"""Lesson 1: angles and coordinate frames. Run this file after filling TODOs.

This lesson moves between the fixed global map and the coordinate frame of the
car. A point does not change physically during a frame conversion; only the
numbers used to describe its direction change.

Relevant syntax:

    vector = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    c = np.cos(angle)
    s = np.sin(angle)
    rotated = matrix @ vector
    direction = np.arctan2(vector[1], vector[0])
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi

NumPy trigonometric functions use radians and work on scalars or arrays.
`np.arctan2(y, x)` takes y first and preserves the quadrant, unlike simply
taking arctan(y/x). Positive rotation and heading are counter-clockwise.

Useful sanity checks: rotating [1, 0] by +pi/2 should point upward, and a global
point directly along the car's heading should have positive local forward
coordinate and nearly zero local left coordinate.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from course_utils import check_close, finish


def wrap_angle(angle: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Wrap angles to the standard interval [-pi, pi).

    Args:
        angle: Scalar angle or ndarray of any shape, in radians.

    Returns:
        Same scalar/shape as angle, with equivalent values in [-pi, pi).

    Angles separated by whole turns are equivalent. Modular arithmetic chooses
    the representative in this interval for either scalars or arrays.
    """
    return np.mod((angle + np.pi), 2 * np.pi) - np.pi


def rotation(angle: float) -> NDArray[np.float64]:
    """Construct the 2D counter-clockwise rotation matrix R(angle).

    Args:
        angle: Scalar rotation angle in radians; positive is counter-clockwise.

    Returns:
        Floating-point ndarray with shape (2, 2).

    The matrix should preserve lengths and rotate positive angles
    counter-clockwise; +pi/2 therefore maps [1, 0] to [0, 1].
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float)


def global_to_local(car_position: ArrayLike, car_heading: float, point: ArrayLike) -> NDArray[np.float64]:
    """Express a global point in the car's coordinate frame.

    Args:
        car_position: Array-like [x, y] with shape (2,), in metres.
        car_heading: Scalar global heading in radians.
        point: Array-like global [x, y] with shape (2,), in metres.

    Returns:
        ndarray [forward, left] with shape (2,), relative to the car.

    Translation forms the car-to-point displacement. Applying the inverse
    heading rotation expresses that vector in the moving car frame.
    """
    return rotation(-car_heading) @ (np.asarray(point, dtype = float) - np.asarray(car_position, dtype = float))



def target_heading(car_position: ArrayLike, point: ArrayLike) -> float:
    """Calculate the global direction from a position toward a point.

    Args:
        car_position, point: Array-like [x, y] values with shape (2,).

    Returns:
        Scalar heading in radians, in atan2's interval [-pi, pi].

    The displacement determines the direction; atan2 uses its y and x
    components to preserve the correct quadrant.
    """
    displacement = np.asarray(point, dtype = float) - np.asarray(car_position, dtype = float)
    return np.arctan2(displacement[1], displacement[0])


def run_checks():
    print("Lesson 1 checks")
    check_close("wrap_angle(): wraps 3pi to -pi", wrap_angle(3 * np.pi), -np.pi)
    check_close("wrap_angle(): wraps negative angles", wrap_angle(-1.5 * np.pi), 0.5 * np.pi)
    check_close("wrap_angle(): supports whole arrays", wrap_angle(np.array([0, 2*np.pi, 7*np.pi/4])), [0, 0, -np.pi/4])
    check_close("rotation(): +pi/2 maps +x to +y", rotation(np.pi/2) @ np.array([1, 0]), [0, 1])
    car = np.array([1.0, 2.0])
    point = np.array([5.0, 4.0])
    check_close("global_to_local(): heading zero needs only translation", global_to_local(car, 0, point), [4, 2])
    check_close("global_to_local(): heading pi/2 rotates by -pi/2", global_to_local(car, np.pi/2, point), [2, -4])
    check_close("target_heading(): uses atan2(delta_y, delta_x)", target_heading(car, point), np.arctan2(2, 4))
    finish(1, "you can describe targets relative to the moving car")


if __name__ == "__main__":
    run_checks()
