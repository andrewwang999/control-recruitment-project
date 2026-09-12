"""Lesson 0: the NumPy needed for this controls project.

Fill each TODO and run:
    python3 exercises/lesson_00_numpy.py

This lesson focuses on arrays, shapes, slicing, vectorization, broadcasting,
reductions, masks, and clipping. Those are the tools used by later lessons.

NUMPY TOOLBOX

The examples below use unrelated sample data. They show the language needed by
the exercises without giving the exercise implementations.

1. Creating and accepting arrays

    vector = np.array([1, 2, 3], dtype=float)
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    zeros = np.zeros(5)
    grid = np.zeros((4, 2))

`np.array` constructs a new ndarray. `np.asarray` is especially useful at a
function boundary because it accepts either a list or an existing ndarray:

    data = np.asarray(user_input, dtype=float)

Use `.copy()` when the function will modify the array but must not modify its
caller's data:

    editable = np.asarray(user_input, dtype=float).copy()

2. Inspecting shape and type

    data.shape       # for example (5,), (N, 2), or (5, N)
    data.ndim        # number of axes
    data.dtype       # stored numeric type
    len(data)        # number of entries along axis 0
    data.size        # total number of scalar values on all axes

For a path shaped `(N, 2)`, `len(path)` is N but `path.size` is 2*N. Path
indices select rows, so the distinction matters.

3. Indexing and slicing

    vector[0]        # first scalar
    vector[-1]       # last scalar
    vector[1:3]      # indices 1 and 2; stop index 3 is excluded
    matrix[2]        # complete row 2
    matrix[:, 0]     # column 0 from every row
    matrix[:2, :]    # first two rows, all columns

Indexing usually removes an axis; slicing usually preserves it. For example,
`vector[0]` is a scalar while `vector[:1]` has shape `(1,)`.

4. Element-wise arithmetic

Same-shaped arrays operate component by component:

    doubled = vector * 2
    squared = vector ** 2
    difference = end - start
    roots = np.sqrt(positive_values)
    magnitudes = np.abs(signed_values)

Useful element-wise functions include `np.sin`, `np.cos`, `np.tan`, `np.sqrt`,
`np.abs`, `np.minimum`, and `np.maximum`. These normally preserve shape.

5. Broadcasting

Broadcasting applies a smaller compatible array across a larger one. For an
`(N, 2)` points array and a `(2,)` position:

    points = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
    position = np.array([1, 2], dtype=float)
    offsets = points - position       # shape (N, 2)

The same position is subtracted from every row. This replaces a loop over path
points and is central to nearest-point calculations.

6. Axes and reductions

A reduction turns several values into fewer values:

    np.min(values)
    np.max(values)
    np.mean(values)
    np.sum(values)
    np.argmin(values)                 # index, not the minimum value

For an `(N, 2)` array, `axis=1` means reduce across the two columns within each
row, producing `(N,)`. `axis=0` means reduce down the N rows, producing `(2,)`:

    row_lengths = np.linalg.norm(offsets, axis=1)  # one length per point
    column_means = np.mean(points, axis=0)          # mean x and mean y

If an axis is confusing, write the input and expected output shapes beside the
line before choosing it.

7. Norms and distances

`np.linalg.norm` computes Euclidean magnitude:

    length = np.linalg.norm(np.array([3, 4]))       # scalar 5
    lengths = np.linalg.norm(offsets, axis=1)       # shape (N,)

The first measures one vector. The second measures every row independently.

8. Transpose and matrix multiplication

    transposed = matrix.T
    result = matrix @ vector

`*` is element-wise multiplication; `@` is matrix multiplication. A standard
rotation formula assumes column vectors (`R @ point`). If points are stored as
rows, the equivalent batch operation is:

    rotated_points = points @ R.T

A simulator history shaped `(5, N)` can become N position rows by selecting
the x/y rows and transposing the result.

9. Clipping, masks, and conditional selection

    bounded = np.clip(values, lower, upper)
    mask = values > threshold                 # Boolean array
    selected = values[mask]                   # values where mask is True
    indices = np.flatnonzero(mask)             # indices where mask is True
    choices = np.where(mask, when_true, when_false)

`np.where(mask)` can also return matching indices, while the three-argument
form chooses values element-wise.

10. Rolling and closed paths

`np.roll` shifts values and wraps entries around the ends:

    next_rows = np.roll(points, -1, axis=0)
    previous_rows = np.roll(points, 1, axis=0)

This is useful when every closed-path point needs its next or previous neighbor.
For a single integer index, wraparound uses modulo: `(index + 1) % len(path)`.

11. Scalar results and debugging

NumPy reductions may return NumPy scalar types. `int(value)` or `float(value)`
can make an ordinary Python scalar when an interface requires one.

When an operation behaves unexpectedly, inspect:

    print(type(value))
    print(np.asarray(value).shape)
    print(np.asarray(value).dtype)
    print(value)

The central habit for this course is to predict the shape before each
operation and verify that the resulting shape matches its physical meaning.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from course_utils import check_close, check_true, finish


# ---------------------------------------------------------------------------
# Part A: constructing and inspecting state vectors
# ---------------------------------------------------------------------------

def make_state(x: float, y: float, heading: float, speed: float, steering: float) -> NDArray[np.float64]:
    """Construct one vehicle state.

    Parameters:
        x, y: Global position in metres (scalars).
        heading: Vehicle heading in radians (scalar).
        speed: Forward speed in m/s (scalar).
        steering: Front-wheel steering angle in radians (scalar).

    Returns:
        Floating-point ndarray with shape (5,), ordered
        [x, y, heading, speed, steering].

    The state is a numeric vector, so preserve the documented order and use a
    floating-point ndarray rather than a plain Python list.
    """
    x = np.array([x, y, heading, speed, steering])
    return x


def position_from_state(state: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract position from a vehicle state.

    Args:
        state: ndarray with shape (5,), ordered [x, y, heading, speed, steering].

    Returns:
        ndarray with shape (2,), containing [x, y].

    Position occupies the first two entries, making this a slicing operation.
    """
    return state[:2]


def speed_and_steering(state: NDArray[np.floating]) -> tuple[float, float]:
    """Extract the final two dynamic quantities from a state.

    Args:
        state: ndarray with shape (5,).

    Returns:
        Pair (speed, steering), both scalars. Remember that state indices start
        at zero, so these occupy indices 3 and 4.

    Indexing syntax: value = array[index]. Multiple values can be returned as
    a tuple with `return first, second`.
    """
    return state[3], state[4]


# ---------------------------------------------------------------------------
# Part B: vector arithmetic, norms, and reductions
# ---------------------------------------------------------------------------

def displacement(start: ArrayLike, end: ArrayLike) -> NDArray[np.float64]:
    """Return the vector pointing from start to end.

    Args:
        start, end: Array-like 2D points with shape (2,).

    Returns:
        ndarray with shape (2,), equal to end - start.

    Treat both inputs as numeric arrays; direction is encoded by end - start.
    """
    return np.asarray(end) - np.asarray(start)


def distance(a: ArrayLike, b: ArrayLike) -> float:
    """Return Euclidean distance between two 2D points.

    Args:
        a, b: Array-like points with shape (2,).

    Returns:
        Nonnegative scalar distance in metres.

    This is the Euclidean norm of the displacement between the points.
    """
    return np.linalg.norm(displacement(np.asarray(a), np.asarray(b)))


def nearest_point_index(position: ArrayLike, points: ArrayLike) -> int:
    """Return the integer index of the closest row in an (N, 2) points array.

    Args:
        position: Array-like point with shape (2,).
        points: ndarray with shape (N, 2); every row is one candidate point.

    Returns:
        Integer in [0, N), identifying a row of points.

    Concept: broadcasting forms one offset per candidate, row-wise norms form
    one distance per candidate, and the minimum identifies the closest row.
    """
    offsets = np.asarray(points) - np.asarray(position)
    distances = np.linalg.norm(offsets, axis=1)
    return np.argmin(distances)
    


def tracking_error_stats(position: ArrayLike, points: ArrayLike) -> tuple[float, float, float]:
    """Summarize distances from one position to many points.

    Args:
        position: Array-like point with shape (2,).
        points: ndarray with shape (N, 2).

    Returns:
        Tuple (minimum, mean, maximum), each a scalar distance.

    Concept: form the shape-(N,) row-wise distances, then summarize that same
    array with three reductions.
    """
    offsets = np.asarray(points) - np.asarray(position)
    distances = np.linalg.norm(offsets, axis=1)
    return (np.min(distances), np.mean(distances), np.max(distances))


# ---------------------------------------------------------------------------
# Part C: shapes, columns, and matrix multiplication
# ---------------------------------------------------------------------------

def states_to_positions(states: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract positions from a state history with shape (5, N).

    Args:
        states: ndarray with shape (5, N), with variables in rows and time in
        columns—the layout returned by Simulator.get_results().

    Returns:
        ndarray with shape (N, 2): one [x, y] row per timestamp.

    The input stores variables in rows, but the output stores one position per
    row; extracting x/y and transposing converts between those conventions.
    """
    positions = states[:2]
    return positions.T


def rotate_points(points: ArrayLike, matrix: ArrayLike) -> NDArray[np.float64]:
    """Rotate every row of an (N, 2) points array by a 2x2 matrix.

    Args:
        points: Array-like collection with shape (N, 2), one point per row.
        matrix: Rotation matrix with shape (2, 2).

    Returns:
        ndarray with shape (N, 2), preserving one rotated point per row.

    For column vectors the formula is R @ p. With points stored as rows, an
    equivalent vectorized expression is `points @ matrix.T`. Here `@` means
    matrix multiplication and `.T` transposes the matrix.
    """
    return points[:2] @ matrix.T


# ---------------------------------------------------------------------------
# Part D: clipping, masks, and conditional array operations
# ---------------------------------------------------------------------------

def clip_control(control: ArrayLike) -> NDArray[np.float64]:
    """Return a new [acceleration, steering_rate] array within project limits.

    Args:
        control: Array-like [acceleration, steering_rate] with shape (2,).

    Returns:
        New floating-point ndarray with shape (2,).

    Limits are acceleration [-10, 4] and steering rate [-1, 1]. Do not mutate
    the caller's input. `np.asarray(control, dtype=float).copy()` creates an
    independent floating-point array before individual entries are clipped.
    """
    result = np.asarray(control, dtype=float).copy()
    result[0] = np.clip(result[0], -10, 4)
    result[1] = np.clip(result[1], -1, 1)
    return result

def violation_indices(accelerations: ArrayLike, limit: float = 12.0) -> NDArray[np.intp]:
    """Return integer indices whose acceleration is strictly above limit.

    Args:
        accelerations: Array-like sequence with shape (N,).
        limit: Maximum allowed value; equality is safe.

    Returns:
        Integer ndarray with shape (K,), containing indices of values > limit.

    A Boolean comparison marks violations; return the positions where that
    mask is true.
    """
    a = np.asarray(accelerations, dtype=float)
    mask = a > limit
    return np.where(mask)[0]


def safe_corner_speeds(curvatures: ArrayLike, lateral_limit: float = 10.0, cap: float = 20.0) -> NDArray[np.float64]:
    """Vectorized preview of Lesson 5's speed formula.

    Args:
        curvatures: Array-like signed curvatures with shape (N,), in 1/m.
        lateral_limit: Allowed lateral acceleration in m/s^2.
        cap: Maximum target speed on nearly straight sections, in m/s.

    Returns:
        ndarray with shape (N,), one safe speed per curvature.

    Greater curvature requires lower speed because lateral acceleration scales
    with speed squared times curvature. Use curvature magnitude, protect the
    near-zero case, and enforce the straight-line speed cap element-wise.
    """
    c = np.asarray(curvatures, dtype = float)
    c = np.abs(c)
    c = np.maximum(c, 1e-6)
    c = lateral_limit / c
    c = np.sqrt(c)
    c = np.clip(c, 0, cap)
    return c


def run_checks():
    print("Lesson 0 checks")

    state = make_state(1, 2, np.pi/2, 8, -.1)
    check_true("make_state(): returns a NumPy array", isinstance(state, np.ndarray))
    check_true("make_state(): shape (5,) and floating dtype", state.shape == (5,) and np.issubdtype(state.dtype, np.floating))
    check_close("make_state(): preserves parameter order", state, [1, 2, np.pi/2, 8, -.1])
    check_close("position_from_state(): extracts [x, y]", position_from_state(state), [1, 2])
    speed, steering = speed_and_steering(state)
    check_close("speed_and_steering(): returns indices 3 and 4", [speed, steering], [8, -.1])

    check_close("displacement(): accepts lists and computes end - start", displacement([1, 2], [4, 6]), [3, 4])
    check_close("distance(): Euclidean 3-4-5 triangle", distance([1, 2], [4, 6]), 5)
    points = np.array([[5., 5.], [1., 1.], [3., 4.], [-2., 0.]])
    check_true("nearest_point_index(): returns the closest row index", nearest_point_index([0, 0], points) == 1)
    check_close(
        "tracking_error_stats(): returns min, mean, max in order",
        tracking_error_stats([0, 0], points),
        [np.sqrt(2), (np.sqrt(50)+np.sqrt(2)+5+2)/4, np.sqrt(50)],
    )

    history = np.array([
        [0, 1, 2],      # x over time
        [4, 5, 6],      # y over time
        [0, 0, 0],      # heading
        [1, 2, 3],      # speed
        [0, .1, .2],    # steering
    ], dtype=float)
    positions = states_to_positions(history)
    check_true("states_to_positions(): converts (5, N) to (N, 2)", positions.shape == (3, 2))
    check_close("states_to_positions(): pairs x and y at each time", positions, [[0, 4], [1, 5], [2, 6]])
    quarter_turn = np.array([[0., -1.], [1., 0.]])
    check_close("rotate_points(): rotates every row counter-clockwise", rotate_points([[1, 0], [0, 2]], quarter_turn), [[0, 1], [-2, 0]])

    original = np.array([9., -3.])
    clipped = clip_control(original)
    check_close("clip_control(): applies separate component limits", clipped, [4, -1])
    check_close("clip_control(): does not mutate its input", original, [9, -3])
    check_close("violation_indices(): finds values strictly above 12", violation_indices([4, 12, 12.01, 8, 14]), [2, 4])
    check_close("safe_corner_speeds(): vectorized curvature formula", safe_corner_speeds([0, .1, .4]), [20, 10, 5])

    # A small payoff: use broadcasting to identify the closest cone.
    car_position = position_from_state(state)
    cone_index = nearest_point_index(car_position, points)
    cone_distance = distance(car_position, points[cone_index])
    finish(0, f"your vectorized geometry found cone {cone_index} at {cone_distance:.3f} m")


if __name__ == "__main__":
    run_checks()
