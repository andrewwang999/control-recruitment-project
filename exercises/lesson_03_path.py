"""Lesson 3: sampled paths, tangents, curvature, and progress.

The path is an ordered `(N, 2)` array with one [x, y] point per row. Ordering
defines forward travel, and the last row connects back to row zero.

Relevant syntax:

    offsets = candidate_points - position
    distances = np.linalg.norm(offsets, axis=1)
    local_choice = int(np.argmin(distances))

    next_points = np.roll(path, -1, axis=0)
    previous_points = np.roll(path, 1, axis=0)
    tangents = next_points - previous_points
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])

    wrapped_index = (index + 1) % len(path)
    candidates = np.arange(start, stop) % len(path)

`axis=0` rolls complete path rows. `axis=1` in a norm reduces each [x, y]
row to one length. `array[:, 0]` and `array[:, 1]` select all x and all y
components respectively.

When searching only candidate rows, `np.argmin` returns an index into the
candidate-distance array, not automatically an index into the full path. Keep
those two index spaces distinct. When differencing headings, wrap the angular
difference before dividing by distance so crossing -pi/pi does not look like a
nearly complete revolution.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from course_utils import check_close, check_true, circle_path, finish


def nearest_path_index(
    position: ArrayLike,
    path: ArrayLike,
    previous_index: int | None = None,
    search_window: int = 40,
) -> int:
    """Return index of the closest path point.

    Args:
        position: Array-like [x, y] with shape (2,).
        path: ndarray with shape (N, 2), one ordered path point per row.
        previous_index: Previously selected path index, or None on first call.
        search_window: Number of indices to consider on each side of previous.

    Returns:
        Integer path-row index in [0, N).

    On the first call, compare against the full path. Later calls use a wrapped
    window around the previous index so nearby track sections do not cause
    discontinuous localization jumps. Row-wise distances identify the nearest
    candidate, which must map back to its actual path index.
    """
    position = np.asarray(position, dtype = float)
    path = np.asarray(path, dtype = float)
    offsets = path - position
    distances = np.linalg.norm(offsets, axis = 1)
    return np.argmin(distances)


def path_headings(path: ArrayLike) -> NDArray[np.float64]:
    """Estimate tangent heading at every point of a closed path.

    Args:
        path: ndarray with shape (N, 2), ordered around a closed loop.

    Returns:
        ndarray with shape (N,), one heading in radians per path point.

    A centered difference between neighboring wrapped points approximates each
    tangent; its y and x components determine the heading through atan2.
    """
    path = np.asarray(path, dtype = float)
    next = np.roll(path, -1, axis=0)
    previous = np.roll(path, 1, axis=0)
    displacements = next - previous
    headings = np.arctan2(displacements[:, 1], displacements[:, 0])
    return headings

def path_curvature(path: ArrayLike) -> NDArray[np.float64]:
    """Estimate signed curvature d(heading)/ds at every point.

    Args:
        path: ndarray with shape (N, 2), ordered around a closed loop.

    Returns:
        ndarray with shape (N,), signed curvature in 1/m. Counter-clockwise
        circles should have positive curvature.

    Curvature is heading change per metre. Use centered neighboring values and
    wrap heading differences across the -pi/pi boundary before dividing by the
    corresponding physical distance.
    """
    
    headings = path_headings(path)
    next = np.roll(headings, -1, axis=0)
    previous = np.roll(headings, 1, axis=0)
    change = (next - previous + np.pi) % (2 * np.pi) - np.pi
    path = np.asarray(path, dtype = float)
    next_path = np.roll(path, -1, axis = 0)
    previous_path = np.roll(path, 1, axis=0)
    distance = np.linalg.norm(next_path - previous_path, axis=1)
    return change / distance

def advance_index(index: int, distance: float, path: ArrayLike) -> int:
    """Walk forward along a closed sampled path and return the first index at
    least `distance` metres ahead.

    Args:
        index: Starting row index.
        distance: Nonnegative lookahead distance in metres.
        path: ndarray with shape (N, 2), representing a closed path.

    Returns:
        Integer path index reached after accumulating at least distance metres.

    Follow successive wrapped segments and accumulate their physical lengths.
    The returned row is the first one whose accumulated distance meets the
    requested lookahead.
    """
    current = index
    path = np.asarray(path, dtype = float)
    total_distance = 0

    while total_distance < distance:
        previous_point = path[current]
        current = (current + 1) % len(path)
        current_point = path[current]
        total_distance += np.linalg.norm(current_point - previous_point)

    return current



def run_checks():
    print("Lesson 3 checks")
    path = circle_path(radius=10, count=1000)
    check_true("nearest_path_index(): finds top of sampled circle", nearest_path_index([0, 10.1], path) in range(248, 253))
    check_true("nearest_path_index(): search window wraps past index N-1", nearest_path_index([10.1, -.1], path, 995, 20) in list(range(0, 5)) + list(range(990, 1000)))
    headings = path_headings(path)
    check_close("path_headings(): circle start tangent points upward", headings[0], np.pi/2, atol=2e-3)
    curvature = path_curvature(path)
    check_close("path_curvature(): 10 m CCW circle has +0.1 1/m", np.mean(curvature), .1, atol=2e-4)
    j = advance_index(0, 5.0, path)
    check_true("advance_index(): walks approximately 5 metres", 78 <= j <= 82, f"Expected an index from 78 through 82; got {j}.")
    finish(3, f"your estimated 10 m circle curvature is {np.mean(curvature):.4f} 1/m")


if __name__ == "__main__":
    run_checks()
