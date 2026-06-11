"""
Custom environment wrapper for GCBF+.

Subclasses SingleIntegrator with three additions:
  1. Optional fixed map (obstacles) and/or fixed task (agent starts + goals).
  2. Continuous goal reassignment: each agent that comes within
     `goal_reach_threshold` of its current goal is immediately assigned a new
     random goal. The episode ends only when the time budget is exhausted
     (default 20 s → 667 steps at dt=0.03 s).
  3. Dynamic obstacles: n_dyn_obs small rectangles that move like u_ref
     (LQR toward a random goal) plus Gaussian velocity noise.  When a dynamic
     obstacle reaches its goal it is assigned a new random one.

Usage
-----
Fixed obstacles, random agents/goals, continuous reassignment, 3 dynamic obstacles:

    env = CustomSingleIntegrator(
        num_agents=4,
        area_size=4.0,
        map_cfg=MapConfig(
            obs_pos=[[1.0, 1.0], [2.5, 0.5]],
            obs_w=[0.4, 0.3],
            obs_h=[0.4, 0.6],
        ),
        n_dyn_obs=3,
    )

Fixed obstacles AND fixed starting task, continuous reassignment:

    env = CustomSingleIntegrator(
        num_agents=2,
        area_size=4.0,
        map_cfg=MapConfig(...),
        task_cfg=TaskConfig(
            agent_pos=[[0.2, 0.2], [0.2, 0.8]],
            goal_pos=[[3.8, 3.8], [3.8, 3.2]],
        ),
    )

CLI (via test.py):

    python test.py --env CustomSingleIntegrator --area-size 4.0 \\
        --path ./pretrained/SingleIntegrator/gcbf+ --epi 5
"""

from __future__ import annotations

import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .single_integrator import SingleIntegrator
from .obstacle import Rectangle
from .utils import get_node_goal_rng, inside_obstacles
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Reward, State


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MapConfig:
    """Fixed obstacle layout.

    All lists must have the same length N (number of obstacles).

    Attributes
    ----------
    obs_pos   : (N, 2) obstacle centre positions [[x, y], ...]
    obs_w     : (N,)   obstacle widths
    obs_h     : (N,)   obstacle heights
    obs_theta : (N,)   obstacle rotation angles in radians (default: all 0)
    """
    obs_pos: List[List[float]]
    obs_w: List[float]
    obs_h: List[float]
    obs_theta: List[float] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.obs_pos)
        assert len(self.obs_w) == n, "obs_w length must match obs_pos"
        assert len(self.obs_h) == n, "obs_h length must match obs_pos"
        if not self.obs_theta:
            self.obs_theta = [0.0] * n
        assert len(self.obs_theta) == n, "obs_theta length must match obs_pos"


@dataclass
class TaskConfig:
    """Fixed agent start positions and initial goal positions.

    Attributes
    ----------
    agent_pos : (num_agents, 2) start positions [[x, y], ...]
    goal_pos  : (num_agents, 2) initial goal positions [[x, y], ...]
    """
    agent_pos: List[List[float]]
    goal_pos: List[List[float]]

    def __post_init__(self):
        assert len(self.agent_pos) == len(self.goal_pos), \
            "agent_pos and goal_pos must have the same length"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_obstacles(obs1: Rectangle, obs2: Rectangle) -> Rectangle:
    """Concatenate two batched Rectangle structs along the obstacle axis."""
    return Rectangle(
        type=jnp.concatenate([obs1.type, obs2.type], axis=0),
        center=jnp.concatenate([obs1.center, obs2.center], axis=0),
        width=jnp.concatenate([obs1.width, obs2.width], axis=0),
        height=jnp.concatenate([obs1.height, obs2.height], axis=0),
        theta=jnp.concatenate([obs1.theta, obs2.theta], axis=0),
        points=jnp.concatenate([obs1.points, obs2.points], axis=0),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_DEFAULT_MAX_STEP = int(60.0 / 0.03)  # 2000 steps = 60 s at dt=0.03


class CustomSingleIntegrator(SingleIntegrator):
    """SingleIntegrator with optional fixed map/task, continuous goal
    reassignment, and randomly moving dynamic obstacles.

    Parameters
    ----------
    map_cfg : MapConfig or None
        Fixed obstacle layout. If None, obstacles are sampled randomly each
        episode (controlled by params['n_obs']).
    task_cfg : TaskConfig or None
        Fixed starting agent/goal positions. If None, positions are sampled
        randomly each episode with collision avoidance.
    goal_reach_threshold : float or None
        Distance at which an agent *or* dynamic obstacle is considered to have
        reached its goal and receives a new one.
        Defaults to 4 × car_radius.
    n_dyn_obs : int
        Number of randomly-moving dynamic obstacles (default 3).
    dyn_obs_size : float
        Side length (width = height) of each dynamic obstacle rectangle.
    dyn_obs_speed_noise : float
        Std-dev of Gaussian noise added to dynamic obstacle velocity at each
        step (same units as action, i.e. m/s before clipping).
    All other parameters are forwarded to SingleIntegrator.
    """

    class EnvState(SingleIntegrator.EnvState.__class__.__bases__[0]):  # NamedTuple
        agent: State
        goal: State
        obstacle: object
        rng_key: Array
        dyn_obs_pos: Array
        dyn_obs_goal: Array

    from typing import NamedTuple as _NT

    class EnvState(_NT):  # type: ignore[no-redef]
        agent: State
        goal: State
        obstacle: object       # Rectangle batch (n_fixed + n_dyn,)
        rng_key: Array         # shape (2,) uint32
        dyn_obs_pos: Array     # (n_dyn_obs, 2)
        dyn_obs_goal: Array    # (n_dyn_obs, 2)

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        map_cfg: Optional[MapConfig] = None,
        task_cfg: Optional[TaskConfig] = None,
        goal_reach_threshold: Optional[float] = None,
        max_step: int = _DEFAULT_MAX_STEP,
        max_travel: Optional[float] = None,
        dt: float = 0.03,
        params: Optional[dict] = None,
        n_dyn_obs: int = 3,
        dyn_obs_size: float = 0.15,
        dyn_obs_speed_noise: float = 0.1,
    ):
        if params is None:
            params = dict(SingleIntegrator.PARAMS)

        if map_cfg is not None:
            params = dict(params)
            params["n_obs"] = len(map_cfg.obs_pos)

        super().__init__(
            num_agents=num_agents,
            area_size=area_size,
            max_step=max_step,
            max_travel=max_travel,
            dt=dt,
            params=params,
        )

        self._map_cfg = map_cfg
        self._task_cfg = task_cfg
        self._goal_reach_threshold = (
            goal_reach_threshold
            if goal_reach_threshold is not None
            else 4.0 * self._params["car_radius"]
        )

        self._n_dyn_obs = n_dyn_obs
        self._dyn_obs_size = dyn_obs_size
        self._dyn_obs_speed_noise = dyn_obs_speed_noise
        # Number of static obstacles (fixed at construction time).
        self._n_fixed_obs = self._params["n_obs"]

        # Pre-build fixed obstacle arrays once; reused every reset().
        if map_cfg is not None:
            self._fixed_obstacles = self.create_obstacles(
                jnp.array(map_cfg.obs_pos, dtype=jnp.float32),
                jnp.array(map_cfg.obs_w, dtype=jnp.float32),
                jnp.array(map_cfg.obs_h, dtype=jnp.float32),
                jnp.array(map_cfg.obs_theta, dtype=jnp.float32),
            )
        else:
            self._fixed_obstacles = None

        if task_cfg is not None:
            assert len(task_cfg.agent_pos) == num_agents, (
                f"task_cfg has {len(task_cfg.agent_pos)} agents but env has {num_agents}"
            )
            self._fixed_states = jnp.array(task_cfg.agent_pos, dtype=jnp.float32)
            self._fixed_goals = jnp.array(task_cfg.goal_pos, dtype=jnp.float32)
        else:
            self._fixed_states = None
            self._fixed_goals = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # --- static obstacles ---
        if self._fixed_obstacles is not None:
            static_obstacles = self._fixed_obstacles
        else:
            n_rng_obs = self._params["n_obs"]
            obstacle_key, key = jr.split(key)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key)
            obs_len = jr.uniform(
                length_key,
                (n_rng_obs, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            static_obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # --- agent start positions and initial goals ---
        if self._fixed_states is not None and self._fixed_goals is not None:
            states = self._fixed_states
            goals = self._fixed_goals
        else:
            pos_key, key = jr.split(key)
            states, goals = get_node_goal_rng(
                pos_key,
                self.area_size,
                2,
                static_obstacles,
                self.num_agents,
                4 * self.params["car_radius"],
                self.max_travel,
            )

        # --- dynamic obstacles ---
        if self._n_dyn_obs > 0:
            dyn_pos_key, key = jr.split(key)
            dyn_obs_pos = jr.uniform(
                dyn_pos_key, (self._n_dyn_obs, 2), minval=0.0, maxval=self.area_size
            )
            dyn_goal_key, key = jr.split(key)
            dyn_obs_goal = jr.uniform(
                dyn_goal_key, (self._n_dyn_obs, 2), minval=0.0, maxval=self.area_size
            )
            dyn_obstacles = self.create_obstacles(
                dyn_obs_pos,
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.zeros(self._n_dyn_obs),
            )
            obstacles = _concat_obstacles(static_obstacles, dyn_obstacles)
        else:
            dyn_obs_pos = jnp.zeros((0, 2), dtype=jnp.float32)
            dyn_obs_goal = jnp.zeros((0, 2), dtype=jnp.float32)
            obstacles = static_obstacles

        step_key, _ = jr.split(key)
        env_states = self.EnvState(states, goals, obstacles, step_key, dyn_obs_pos, dyn_obs_goal)
        return self.get_graph(env_states)

    # ------------------------------------------------------------------
    # step — continuous goal reassignment + dynamic obstacle motion
    # ------------------------------------------------------------------

    def step(
        self,
        graph: GraphsTuple,
        action: Action,
        get_eval_info: bool = False,
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle
        rng_key = graph.env_states.rng_key
        dyn_obs_pos = graph.env_states.dyn_obs_pos
        dyn_obs_goal = graph.env_states.dyn_obs_goal

        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)

        # Reward and cost (same as base class)
        done = jnp.array(False)
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        # --- continuous agent goal reassignment ---
        key = rng_key
        key, use_key = jr.split(key)
        dist_to_goal = jnp.linalg.norm(next_agent_states - goals, axis=1)
        reached = dist_to_goal < self._goal_reach_threshold
        per_agent_keys = jr.split(use_key, self.num_agents)
        new_goals_cand = jax.vmap(
            lambda k: jr.uniform(k, (self.state_dim,), minval=0.0, maxval=self.area_size)
        )(per_agent_keys)
        updated_goals = jnp.where(reached[:, None], new_goals_cand, goals)

        # --- dynamic obstacle motion ---
        if self._n_dyn_obs > 0:
            # u_ref-style control toward each obstacle's goal
            error = dyn_obs_goal - dyn_obs_pos  # (n_dyn, 2)
            norm = jnp.linalg.norm(error, axis=-1, keepdims=True) + 1e-6
            error_max = jnp.abs(error / norm * self._params["comm_radius"])
            error_clipped = jnp.clip(error, -error_max, error_max)
            dyn_vel = self.clip_action(error_clipped @ self._K.T)

            # Add Gaussian speed noise
            key, noise_key = jr.split(key)
            noise = jr.normal(noise_key, dyn_obs_pos.shape) * self._dyn_obs_speed_noise
            dyn_vel = self.clip_action(dyn_vel + noise)

            # Euler integration, clamped to arena bounds
            new_dyn_pos = jnp.clip(
                dyn_obs_pos + dyn_vel * self._dt, 0.0, self.area_size
            )

            # Goal reassignment when an obstacle reaches its goal
            dyn_dist = jnp.linalg.norm(new_dyn_pos - dyn_obs_goal, axis=-1)
            dyn_reached = dyn_dist < self._goal_reach_threshold
            key, dyn_goal_key = jr.split(key)
            per_obs_keys = jr.split(dyn_goal_key, self._n_dyn_obs)
            new_dyn_goals_cand = jax.vmap(
                lambda k: jr.uniform(k, (2,), minval=0.0, maxval=self.area_size)
            )(per_obs_keys)
            new_dyn_goals = jnp.where(dyn_reached[:, None], new_dyn_goals_cand, dyn_obs_goal)

            # Rebuild dynamic Rectangle objects with updated centers
            new_dyn_obstacles = self.create_obstacles(
                new_dyn_pos,
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.zeros(self._n_dyn_obs),
            )
            fixed_obs = jax.tree_util.tree_map(lambda x: x[: self._n_fixed_obs], obstacles)
            new_obstacles = _concat_obstacles(fixed_obs, new_dyn_obstacles)
        else:
            new_dyn_pos = dyn_obs_pos
            new_dyn_goals = dyn_obs_goal
            new_obstacles = obstacles

        next_state = self.EnvState(
            next_agent_states, updated_goals, new_obstacles, key,
            new_dyn_pos, new_dyn_goals,
        )

        info = {}
        if get_eval_info:
            info["inside_obstacles"] = inside_obstacles(
                agent_states, obstacles, r=self._params["car_radius"]
            )

        return self.get_graph(next_state), reward, cost, done, info


# ---------------------------------------------------------------------------
# Circuit phase constants
# ---------------------------------------------------------------------------

_PHASE_INACTIVE       = 0
_PHASE_PICKUP_DWELL   = 1
_PHASE_TO_DELIVERY    = 2
_PHASE_DELIVERY_DWELL = 3
_PHASE_TO_DROPOFF     = 4
_PHASE_DROPOFF_DWELL  = 5


@dataclass
class CircuitConfig:
    """Circuit layout for logistics simulation.

    Parameters
    ----------
    pickup_pos      : [x, y] where robots spawn and dwell first.
    delivery_pos    : list of k=2 positions robots can be routed to.
    dropoff_pos     : [x, y] where robots dwell last and disappear.
    total_robots    : how many robots complete the circuit before spawning stops.
    spawn_interval  : seconds between consecutive robot spawns.
    dwell_time      : seconds a robot dwells at each station.
    """
    pickup_pos:     List[float]
    delivery_pos:   List[List[float]]
    dropoff_pos:    List[float]
    total_robots:   int   = 10
    spawn_interval: float = 5.0
    dwell_time:     float = 0.0


# ---------------------------------------------------------------------------
# CircuitEnv
# ---------------------------------------------------------------------------

class CircuitEnv(CustomSingleIntegrator):
    """Logistics circuit built on top of CustomSingleIntegrator.

    Each robot slot follows a single circuit:
      spawn at pickup → dwell → travel to a random delivery station →
      dwell → travel to drop-off → dwell → disappear.

    Parameters
    ----------
    num_agents : int
        Maximum robots active in the scene at the same time.
    circuit_cfg : CircuitConfig
        Circuit layout and timing.  If None, taken from params['circuit_cfg'].
    All other keyword arguments are forwarded to CustomSingleIntegrator.
    """

    _DEFAULT_CIRCUIT_CFG = CircuitConfig(
        pickup_pos=[1.0, 1.0],
        delivery_pos=[[3.0, 1.0], [3.0, 3.0]],
        dropoff_pos=[1.0, 3.0],
        total_robots=10,
        spawn_interval=5.0,
        dwell_time=0.0,
    )

    PARAMS = {
        **SingleIntegrator.PARAMS,
        'circuit_cfg': _DEFAULT_CIRCUIT_CFG,
    }

    from typing import NamedTuple as _NT

    class EnvState(_NT):  # type: ignore[no-redef]
        agent:          State
        goal:           State
        obstacle:       object
        rng_key:        Array
        dyn_obs_pos:    Array     # (n_dyn_obs, 2)
        dyn_obs_goal:   Array     # (n_dyn_obs, 2)
        robot_phase:    Array     # (num_agents,) int32
        robot_timer:    Array     # (num_agents,) int32 — steps in current phase
        spawn_timer:    Array     # ()  int32 — steps until next spawn
        robots_spawned: Array     # ()  int32 — total robots spawned so far
        robot_delivery: Array     # (num_agents, 2) — delivery goal locked at spawn

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        circuit_cfg: Optional[CircuitConfig] = None,
        map_cfg: Optional[MapConfig] = None,
        n_dyn_obs: int = 0,
        dyn_obs_size: float = 0.15,
        dyn_obs_speed_noise: float = 0.1,
        max_step: int = _DEFAULT_MAX_STEP,
        max_travel: Optional[float] = None,
        dt: float = 0.03,
        params: Optional[dict] = None,
    ):
        if circuit_cfg is None:
            circuit_cfg = (params or {}).get('circuit_cfg', CircuitEnv._DEFAULT_CIRCUIT_CFG)
        # strip circuit_cfg from params so parent doesn't see it
        if params is not None and 'circuit_cfg' in params:
            params = {k: v for k, v in params.items() if k != 'circuit_cfg'}
        super().__init__(
            num_agents=num_agents,
            area_size=area_size,
            map_cfg=map_cfg,
            n_dyn_obs=n_dyn_obs,
            dyn_obs_size=dyn_obs_size,
            dyn_obs_speed_noise=dyn_obs_speed_noise,
            max_step=max_step,
            max_travel=max_travel,
            dt=dt,
            params=params,
        )
        self._circuit_cfg  = circuit_cfg
        self._pickup       = jnp.array(circuit_cfg.pickup_pos,   dtype=jnp.float32)   # (2,)
        self._delivery     = jnp.array(circuit_cfg.delivery_pos, dtype=jnp.float32)   # (k, 2)
        self._dropoff      = jnp.array(circuit_cfg.dropoff_pos,  dtype=jnp.float32)   # (2,)
        self._garage       = jnp.array([-2.0 * area_size, -2.0 * area_size], dtype=jnp.float32)
        self._dwell_steps  = int(round(circuit_cfg.dwell_time    / dt))
        self._spawn_steps  = int(round(circuit_cfg.spawn_interval / dt))
        self._total_robots = circuit_cfg.total_robots

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # --- obstacles (reuse parent logic) ---
        if self._fixed_obstacles is not None:
            static_obstacles = self._fixed_obstacles
        else:
            n_rng_obs = self._params["n_obs"]
            obstacle_key, key = jr.split(key)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key)
            obs_len = jr.uniform(
                length_key, (n_rng_obs, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            static_obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        if self._n_dyn_obs > 0:
            dyn_pos_key, key = jr.split(key)
            dyn_obs_pos = jr.uniform(dyn_pos_key, (self._n_dyn_obs, 2), minval=0.0, maxval=self.area_size)
            dyn_goal_key, key = jr.split(key)
            dyn_obs_goal = jr.uniform(dyn_goal_key, (self._n_dyn_obs, 2), minval=0.0, maxval=self.area_size)
            dyn_obstacles = self.create_obstacles(
                dyn_obs_pos,
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.zeros(self._n_dyn_obs),
            )
            obstacles = _concat_obstacles(static_obstacles, dyn_obstacles)
        else:
            dyn_obs_pos  = jnp.zeros((0, 2), dtype=jnp.float32)
            dyn_obs_goal = jnp.zeros((0, 2), dtype=jnp.float32)
            obstacles    = static_obstacles

        step_key, _ = jr.split(key)
        n = self.num_agents
        garage_tile = jnp.tile(self._garage, (n, 1))
        pickup_tile = jnp.tile(self._pickup, (n, 1))

        env_states = self.EnvState(
            agent          = garage_tile,
            goal           = pickup_tile,   # overwritten on first spawn
            obstacle       = obstacles,
            rng_key        = step_key,
            dyn_obs_pos    = dyn_obs_pos,
            dyn_obs_goal   = dyn_obs_goal,
            robot_phase    = jnp.zeros(n, dtype=jnp.int32),
            robot_timer    = jnp.zeros(n, dtype=jnp.int32),
            spawn_timer    = jnp.array(0, dtype=jnp.int32),
            robots_spawned = jnp.array(0, dtype=jnp.int32),
            robot_delivery = jnp.tile(self._delivery[0], (n, 1)),
        )
        return self.get_graph(env_states)

    # ------------------------------------------------------------------
    # u_ref — same as parent but with epsilon guard against zero-norm
    # ------------------------------------------------------------------

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal  = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        norm  = jnp.linalg.norm(error, axis=-1, keepdims=True) + 1e-6
        error_max = jnp.abs(error / norm * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    # ------------------------------------------------------------------
    # step — phase state machine + spawn logic
    # ------------------------------------------------------------------

    def step(
        self,
        graph: GraphsTuple,
        action: Action,
        get_eval_info: bool = False,
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        agent_states   = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals          = graph.env_states.goal
        obstacles      = graph.env_states.obstacle
        rng_key        = graph.env_states.rng_key
        dyn_obs_pos    = graph.env_states.dyn_obs_pos
        dyn_obs_goal   = graph.env_states.dyn_obs_goal
        robot_phase    = graph.env_states.robot_phase
        robot_timer    = graph.env_states.robot_timer
        spawn_timer    = graph.env_states.spawn_timer
        robots_spawned = graph.env_states.robots_spawned
        robot_delivery = graph.env_states.robot_delivery

        # Inactive agents don't receive actions; hard-freeze at garage.
        n = self.num_agents
        garage_tile = jnp.tile(self._garage, (n, 1))
        is_inactive = (robot_phase == _PHASE_INACTIVE)
        frozen_action = jnp.where(is_inactive[:, None], jnp.zeros_like(action), self.clip_action(action))
        next_agent_states = self.agent_step_euler(agent_states, frozen_action)
        next_agent_states = jnp.where(is_inactive[:, None], garage_tile, next_agent_states)

        done   = jnp.array(False)
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost   = self.get_cost(graph)

        key = rng_key

        # --- Phase state machine ---
        timer_expired = (robot_timer + 1 >= self._dwell_steps)
        dist_to_goal  = jnp.linalg.norm(next_agent_states - goals, axis=-1)
        arrived       = dist_to_goal < self._goal_reach_threshold

        dropoff_tile = jnp.tile(self._dropoff, (n, 1))
        pickup_tile  = jnp.tile(self._pickup,  (n, 1))

        next_phase = robot_phase
        next_phase = jnp.where((robot_phase == _PHASE_PICKUP_DWELL)   & timer_expired, _PHASE_TO_DELIVERY,    next_phase)
        next_phase = jnp.where((robot_phase == _PHASE_TO_DELIVERY)    & arrived,       _PHASE_DELIVERY_DWELL, next_phase)
        next_phase = jnp.where((robot_phase == _PHASE_DELIVERY_DWELL) & timer_expired, _PHASE_TO_DROPOFF,     next_phase)
        next_phase = jnp.where((robot_phase == _PHASE_TO_DROPOFF)     & arrived,       _PHASE_DROPOFF_DWELL,  next_phase)
        next_phase = jnp.where((robot_phase == _PHASE_DROPOFF_DWELL)  & timer_expired, _PHASE_INACTIVE,       next_phase)

        phase_changed = (next_phase != robot_phase)
        next_timer    = jnp.where(phase_changed, 0, robot_timer + 1)

        # Goals follow phase transitions.  Delivery uses the per-slot locked goal.
        new_goals = goals
        new_goals = jnp.where((next_phase == _PHASE_TO_DELIVERY)[:, None],   robot_delivery, new_goals)
        new_goals = jnp.where((next_phase == _PHASE_TO_DROPOFF)[:, None],    dropoff_tile,   new_goals)
        new_goals = jnp.where((next_phase == _PHASE_INACTIVE)[:, None],      garage_tile,    new_goals)

        # Teleport newly-inactive robots back to garage.
        next_agent_states = jnp.where((next_phase == _PHASE_INACTIVE)[:, None], garage_tile, next_agent_states)

        # --- Spawn logic ---
        new_spawn_timer    = spawn_timer - 1
        should_spawn       = (new_spawn_timer <= 0)
        budget_remaining   = robots_spawned < self._total_robots
        # Only claim a slot that was already inactive BEFORE this step.
        any_inactive       = jnp.any(robot_phase == _PHASE_INACTIVE)
        do_spawn           = should_spawn & any_inactive & budget_remaining
        # Keep timer at 0 until a slot frees; reset to interval once spawn fires.
        new_spawn_timer    = jnp.where(do_spawn, self._spawn_steps, jnp.maximum(new_spawn_timer, 0))
        new_robots_spawned = jnp.where(do_spawn, robots_spawned + 1, robots_spawned)

        slot_scores = jnp.where(robot_phase == _PHASE_INACTIVE, jnp.arange(n), n + 1)
        first_slot  = jnp.argmin(slot_scores)
        spawn_mask  = (jnp.arange(n) == first_slot) & do_spawn   # (n,) bool

        # Pick a random delivery station for the spawning slot and lock it in.
        key, deliv_key = jr.split(key)
        deliv_idx      = jr.randint(deliv_key, (), 0, self._delivery.shape[0])
        new_robot_delivery = jnp.where(spawn_mask[:, None],
                                       jnp.tile(self._delivery[deliv_idx], (n, 1)),
                                       robot_delivery)

        next_phase        = jnp.where(spawn_mask, _PHASE_PICKUP_DWELL, next_phase)
        next_timer        = jnp.where(spawn_mask, 0, next_timer)
        new_goals         = jnp.where(spawn_mask[:, None], pickup_tile, new_goals)
        next_agent_states = jnp.where(spawn_mask[:, None], pickup_tile, next_agent_states)

        # --- Dynamic obstacle motion (only when n_dyn_obs > 0) ---
        if self._n_dyn_obs > 0:
            error     = dyn_obs_goal - dyn_obs_pos
            norm      = jnp.linalg.norm(error, axis=-1, keepdims=True) + 1e-6
            error_max = jnp.abs(error / norm * self._params["comm_radius"])
            dyn_vel   = self.clip_action(jnp.clip(error, -error_max, error_max) @ self._K.T)
            key, noise_key = jr.split(key)
            dyn_vel   = self.clip_action(dyn_vel + jr.normal(noise_key, dyn_obs_pos.shape) * self._dyn_obs_speed_noise)
            new_dyn_pos = jnp.clip(dyn_obs_pos + dyn_vel * self._dt, 0.0, self.area_size)

            dyn_dist  = jnp.linalg.norm(new_dyn_pos - dyn_obs_goal, axis=-1)
            key, dyn_goal_key = jr.split(key)
            per_obs_keys = jr.split(dyn_goal_key, self._n_dyn_obs)
            new_dyn_goals_cand = jax.vmap(
                lambda k: jr.uniform(k, (2,), minval=0.0, maxval=self.area_size)
            )(per_obs_keys)
            new_dyn_goals = jnp.where(
                (dyn_dist < self._goal_reach_threshold)[:, None], new_dyn_goals_cand, dyn_obs_goal
            )
            new_dyn_obstacles = self.create_obstacles(
                new_dyn_pos,
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.full(self._n_dyn_obs, self._dyn_obs_size),
                jnp.zeros(self._n_dyn_obs),
            )
            fixed_obs    = jax.tree_util.tree_map(lambda x: x[:self._n_fixed_obs], obstacles)
            new_obstacles = _concat_obstacles(fixed_obs, new_dyn_obstacles)
        else:
            new_dyn_pos    = dyn_obs_pos
            new_dyn_goals  = dyn_obs_goal
            new_obstacles  = obstacles

        next_state = self.EnvState(
            agent          = next_agent_states,
            goal           = new_goals,
            obstacle       = new_obstacles,
            rng_key        = key,
            dyn_obs_pos    = new_dyn_pos,
            dyn_obs_goal   = new_dyn_goals,
            robot_phase    = next_phase,
            robot_timer    = next_timer,
            spawn_timer    = new_spawn_timer,
            robots_spawned = new_robots_spawned,
            robot_delivery = new_robot_delivery,
        )

        info = {}
        if get_eval_info:
            info["inside_obstacles"] = inside_obstacles(
                agent_states, obstacles, r=self._params["car_radius"]
            )

        return self.get_graph(next_state), reward, cost, done, info

    # ------------------------------------------------------------------
    # Cost — only active robots count
    # ------------------------------------------------------------------

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles    = graph.env_states.obstacle
        is_active    = (graph.env_states.robot_phase > _PHASE_INACTIVE).astype(jnp.float32)

        agent_pos   = agent_states
        dist        = jnp.linalg.norm(
            jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
        )
        dist       += jnp.eye(self.num_agents) * 1e6
        active_pair = is_active[:, None] * is_active[None, :]
        cost  = ((self._params["car_radius"] * 2 > dist) * active_pair).any(axis=1)
        cost  = (cost * is_active).mean()
        cost += (inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"]) * is_active).mean()
        return cost

    # ------------------------------------------------------------------
    # Masks — exclude inactive robots
    # ------------------------------------------------------------------

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        is_active = graph.env_states.robot_phase > _PHASE_INACTIVE
        return jnp.where(is_active, SingleIntegrator.safe_mask(self, graph), True)

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        is_active = graph.env_states.robot_phase > _PHASE_INACTIVE
        return jnp.where(is_active, SingleIntegrator.unsafe_mask(self, graph), False)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        return self.unsafe_mask(graph)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        return jnp.zeros(self.num_agents, dtype=jnp.bool_)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_video(self, rollout, video_path, Ta_is_unsafe=None, viz_opts=None, dpi=100, **kwargs):
        from .plot import render_video_circuit
        render_video_circuit(
            rollout       = rollout,
            video_path    = video_path,
            side_length   = self.area_size,
            n_agent       = self.num_agents,
            n_rays        = self.params["n_rays"],
            r             = self.params["car_radius"],
            pickup_pos    = np.array(self._pickup),
            delivery_pos  = np.array(self._delivery),
            dropoff_pos   = np.array(self._dropoff),
            Ta_is_unsafe  = Ta_is_unsafe,
            viz_opts      = viz_opts,
            dpi           = dpi,
            **kwargs,
        )
