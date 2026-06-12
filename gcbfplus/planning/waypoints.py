"""A* waypoint generation over a traversable-rectangle graph.

All functions are pure Python/NumPy — no JAX dependency.
"""

import heapq
import json
import math
from typing import Optional

import numpy as np


def load_traversable_rects(path: str) -> list:
    """Load the list of rectangle dicts from a traversable-rectangles JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["rectangles"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rects_overlap(r1: dict, r2: dict, tol: float = 0.05) -> bool:
    """True if two axis-aligned rect dicts share area (with tolerance)."""
    return (
        r1["x_min"] < r2["x_max"] + tol
        and r1["x_max"] + tol > r2["x_min"]
        and r1["y_min"] < r2["y_max"] + tol
        and r1["y_max"] + tol > r2["y_min"]
    )


def _build_rect_graph(rects: list) -> list:
    """Build weighted adjacency list over rectangle centers.

    Returns:
        adj: adj[i] = [(j, dist), ...] for each overlapping neighbor j.
    """
    n = len(rects)
    centers = [np.array(r["center"], dtype=float) for r in rects]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _rects_overlap(rects[i], rects[j]):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                adj[i].append((j, d))
                adj[j].append((i, d))
    return adj


def _intersection_center(r1: dict, r2: dict) -> np.ndarray:
    """Center of the axis-aligned intersection of two overlapping rectangles."""
    x_lo = max(r1["x_min"], r2["x_min"])
    x_hi = min(r1["x_max"], r2["x_max"])
    y_lo = max(r1["y_min"], r2["y_min"])
    y_hi = min(r1["y_max"], r2["y_max"])
    return np.array([(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0], dtype=float)


def _find_containing_rect(point: np.ndarray, rects: list, tol: float = 0.1) -> int:
    """Index of the rect whose bounds contain *point* (with tolerance).

    Falls back to the rect with the nearest center if none contains the point.
    """
    px, py = float(point[0]), float(point[1])
    for i, r in enumerate(rects):
        if r["x_min"] - tol <= px <= r["x_max"] + tol and r["y_min"] - tol <= py <= r["y_max"] + tol:
            return i
    # fallback: nearest center
    centers = [np.array(r["center"], dtype=float) for r in rects]
    dists = [np.linalg.norm(c - point) for c in centers]
    return int(np.argmin(dists))


def _astar(adj: list, centers: list, start: int, end: int) -> Optional[list]:
    """A* over the rectangle graph.

    Returns list of rect indices [start, ..., end], or None if unreachable.
    Heuristic: Euclidean distance to end center.
    """
    if start == end:
        return [start]

    end_c = centers[end]
    # heap entries: (f, g, node, path)
    open_heap = [(math.dist(centers[start], end_c), 0.0, start, [start])]
    visited: dict[int, float] = {}

    while open_heap:
        _, g, node, path = heapq.heappop(open_heap)
        if node in visited:
            continue
        visited[node] = g
        if node == end:
            return path
        for neighbor, d in adj[node]:
            if neighbor not in visited:
                ng = g + d
                nf = ng + math.dist(centers[neighbor], end_c)
                heapq.heappush(open_heap, (nf, ng, neighbor, path + [neighbor]))

    return None  # unreachable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_polyline(points: list, spacing: float) -> list:
    """Resample a polyline so consecutive points are ~spacing apart.

    Always ends exactly at points[-1].  Never returns an empty list.
    """
    result = []
    dist_since_last = 0.0

    for i in range(len(points) - 1):
        p0 = np.asarray(points[i], dtype=float)
        p1 = np.asarray(points[i + 1], dtype=float)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len == 0.0:
            continue

        dist_to_next = spacing - dist_since_last
        d = dist_to_next  # position along this segment for the first waypoint

        if d > seg_len:
            dist_since_last += seg_len
        else:
            while d <= seg_len:
                result.append(p0 + (d / seg_len) * seg)
                d += spacing
            dist_since_last = seg_len - (d - spacing)

    end = np.asarray(points[-1], dtype=float)
    if len(result) == 0 or np.linalg.norm(result[-1] - end) > 1e-6:
        result.append(end)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_waypoints(
    start: np.ndarray,
    end: np.ndarray,
    rects: list,
    wp_spacing: float = 1.0,
) -> np.ndarray:
    """Compute waypoints from *start* to *end* via A*, spaced *wp_spacing* apart.

    Resamples the A* path so consecutive waypoints are ~wp_spacing units apart,
    keeping the agent within traversable rectangles.
    Always includes *end* as the last point; never includes *start*.
    Shape is ``(n, 2)`` with ``n >= 1``.

    Falls back to ``[[end]]`` when the graph is disconnected or start == end rect.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if not rects:
        return np.array([end], dtype=np.float32)

    adj = _build_rect_graph(rects)
    centers = [np.array(r["center"], dtype=float) for r in rects]

    s_idx = _find_containing_rect(start, rects)
    e_idx = _find_containing_rect(end, rects)

    path_idxs = _astar(adj, centers, s_idx, e_idx)

    if path_idxs is None:
        return np.array([end], dtype=np.float32)

    # Build the polyline through intersection centers so every segment lies
    # entirely within one convex rectangle (traversable by construction).
    # Segment start→t(0,1): both in rects[path[0]]  ✓
    # Segment t(i,i+1)→t(i+1,i+2): both in rects[path[i+1]]  ✓
    # Segment t(n-1,n)→end: both in rects[path[n]]  ✓
    transitions = [
        _intersection_center(rects[path_idxs[i]], rects[path_idxs[i + 1]])
        for i in range(len(path_idxs) - 1)
    ]
    polyline = [start] + transitions + [end]

    waypoints = _resample_polyline(polyline, wp_spacing)
    return np.array(waypoints, dtype=np.float32)
