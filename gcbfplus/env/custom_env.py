"""
Custom environment wrapper for GCBF+.

Subclasses SingleIntegrator with two additions:
  1. Optional fixed map (obstacles) and/or fixed task (agent starts + goals).
  2. Continuous goal reassignment: each agent that comes within
     `goal_reach_threshold` of its current goal is immediately assigned a new
     random goal. The episode ends only when the time budget is exhausted
     (default 20 s → 667 steps at dt=0.03 s).

Usage
-----
Fixed obstacles, random agents/goals, continuous reassignment:

    env = CustomSingleIntegrator(
        num_agents=4,
        area_size=4.0,
        map_cfg=MapConfig(
            obs_pos=[[1.0, 1.0], [2.5, 0.5]],
            obs_w=[0.4, 0.3],
            obs_h=[0.4, 0.6],
        ),
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

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional

from .single_integrator import SingleIntegrator
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
# Environment
# ---------------------------------------------------------------------------

# 20 s at dt=0.03 s per step
_DEFAULT_MAX_STEP = int(20.0 / 0.03)  # 666


class CustomSingleIntegrator(SingleIntegrator):
    """SingleIntegrator with optional fixed map/task and continuous goal reassignment.

    Parameters
    ----------
    map_cfg : MapConfig or None
        Fixed obstacle layout. If None, obstacles are sampled randomly each
        episode (controlled by params['n_obs']).
    task_cfg : TaskConfig or None
        Fixed starting agent/goal positions. If None, positions are sampled
        randomly each episode with collision avoidance.
    goal_reach_threshold : float or None
        Distance at which an agent is considered to have reached its goal and
        receives a new one. Defaults to 4 × car_radius.
    All other parameters are forwarded to SingleIntegrator.
    """

    # Extended env state that carries a PRNG key so goal resampling is
    # JAX-jittable and compatible with jax.lax.scan inside rollout_fn.
    class EnvState(SingleIntegrator.EnvState.__class__.__bases__[0]):  # NamedTuple
        agent: State
        goal: State
        obstacle: object  # Obstacle (Rectangle)
        rng_key: Array  # shape (2,) uint32 — threaded through every step

    # Re-declare as a proper NamedTuple (the above trick doesn't work cleanly)
    from typing import NamedTuple as _NT

    class EnvState(_NT):  # type: ignore[no-redef]
        agent: State
        goal: State
        obstacle: object
        rng_key: Array

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

        # --- obstacles ---
        if self._fixed_obstacles is not None:
            obstacles = self._fixed_obstacles
            step_key, key = jr.split(key)
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
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
            step_key, key = jr.split(key)

        # --- agent start positions and initial goals ---
        if self._fixed_states is not None and self._fixed_goals is not None:
            states = self._fixed_states
            goals = self._fixed_goals
        else:
            states, goals = get_node_goal_rng(
                key,
                self.area_size,
                2,
                obstacles,
                self.num_agents,
                4 * self.params["car_radius"],
                self.max_travel,
            )

        env_states = self.EnvState(states, goals, obstacles, step_key)
        return self.get_graph(env_states)

    # ------------------------------------------------------------------
    # step — continuous goal reassignment
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

        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)

        # Reward and cost (same as base class)
        done = jnp.array(False)
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        # --- continuous goal reassignment ---
        dist_to_goal = jnp.linalg.norm(next_agent_states - goals, axis=1)  # (n_agents,)
        reached = dist_to_goal < self._goal_reach_threshold               # (n_agents,) bool

        # Draw a fresh candidate goal for every agent; only apply where reached.
        new_key, use_key = jr.split(rng_key)
        per_agent_keys = jr.split(use_key, self.num_agents)
        new_goals = jax.vmap(
            lambda k: jr.uniform(k, (self.state_dim,), minval=0.0, maxval=self.area_size)
        )(per_agent_keys)                                                  # (n_agents, state_dim)
        updated_goals = jnp.where(reached[:, None], new_goals, goals)

        next_state = self.EnvState(next_agent_states, updated_goals, obstacles, new_key)

        info = {}
        if get_eval_info:
            info["inside_obstacles"] = inside_obstacles(
                agent_states, obstacles, r=self._params["car_radius"]
            )

        return self.get_graph(next_state), reward, cost, done, info
