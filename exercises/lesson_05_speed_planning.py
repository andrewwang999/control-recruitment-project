"""Lesson 5: acceleration-circle safety and braking-aware speed profiles.

This lesson separates planning from feedback. Planning creates one desired
speed per path point using future geometry; feedback turns the desired speed at
the current point into an acceleration command.

Relevant array syntax:

    values = np.asarray(values, dtype=float)
    magnitudes = np.abs(values)
    floored = np.maximum(magnitudes, epsilon)
    capped = np.minimum(candidate_values, cap)
    roots = np.sqrt(nonnegative_values)
    profile = np.asarray(raw_limits, dtype=float).copy()

These NumPy functions operate element-wise and preserve array shape. Python's
`min(a, b)` is suitable for two scalars; `np.minimum(a, b)` is the array-aware
element-wise version. Protect square roots and divisors from invalid values.

A backward loop over N path entries can use:

    for i in range(N - 1, -1, -1):
        next_i = (i + 1) % N

The syntax walks from `N-1` through `0`; modulo connects the last point to the
first. The conceptual purpose is to let a slow future corner restrict earlier
speeds, not merely to reverse an array.

For feedback, prefer keyword arguments when several optional parameters are
numeric, for example `function(value, curvature=current_curvature)`. This
makes the physical role explicit and avoids putting a number into the wrong
parameter while still producing syntactically valid code.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from course_utils import check_close, check_true, finish


def curvature_speed_limit(
    curvature: float | NDArray[np.floating],
    lateral_limit: float = 10.0,
    straight_speed: float = 20.0,
    epsilon: float = 1e-6,
) -> float | NDArray[np.float64]:
    """Return min(straight_speed, sqrt(lateral_limit / max(abs(kappa), epsilon))).

    Args:
        curvature: Scalar or ndarray of signed path curvature in 1/m.
        lateral_limit: Positive allowed lateral acceleration in m/s^2.
        straight_speed: Speed cap in m/s when curvature is small.
        epsilon: Positive denominator floor preventing division by zero.

    Returns:
        Scalar if curvature is scalar, otherwise ndarray of the same shape;
        every value is a nonnegative speed in m/s.

    Since lateral acceleration is speed^2*|curvature|, tighter turns imply
    lower safe speeds. The epsilon handles near-straight points and the speed
    cap handles the unbounded straight-line case.
    """
    curvature = np.abs(np.asarray(curvature, dtype=float))
    safe_curvature = np.maximum(curvature, epsilon)
    corner_speed = np.sqrt(lateral_limit / safe_curvature)
    return np.minimum(corner_speed, straight_speed)

def longitudinal_budget(speed: float, curvature: float, total_limit: float = 11.0) -> float:
    """Return remaining |longitudinal acceleration| under the acceleration circle.

    Args:
        speed: Current vehicle speed in m/s.
        curvature: Current/reference path curvature in 1/m.
        total_limit: Conservative combined-acceleration limit in m/s^2.

    Returns:
        Nonnegative scalar maximum magnitude of longitudinal acceleration.

    Turning consumes lateral acceleration equal to speed^2*|curvature|. The
    remaining longitudinal allowance is the other side of the shared
    acceleration circle, with no negative square-root argument.
    """
    lateral = speed**2 * abs(curvature)
    remaining_squared = np.maximum(total_limit**2 - lateral**2, 0.0)
    return float(np.sqrt(remaining_squared))


def braking_pass(
    raw_speed_limits: ArrayLike,
    segment_lengths: ArrayLike,
    braking: float = 8.0,
    passes: int = 2,
) -> NDArray[np.float64]:
    """Return a braking-feasible speed profile for a closed path.

    Args:
        raw_speed_limits: Array-like shape (N,), one local speed cap per point.
        segment_lengths: Array-like shape (N,); entry i is distance from point
            i to point (i+1) % N.
        braking: Positive available deceleration magnitude in m/s^2.
        passes: Number of complete backward sweeps around the closed path.

    Returns:
        New floating-point ndarray with shape (N,), never exceeding raw limits.

    A point is feasible only if the car can brake from its speed to the next
    point's speed over the connecting segment. Propagate this constraint
    backward around the closed loop using
    v[i] <= sqrt(v[next]^2 + 2*braking*segment_length[i]). Return a copy rather
    than changing the raw limits.
    """
    profile = np.asarray(raw_speed_limits, dtype=float).copy()
    segment_lengths = np.asarray(segment_lengths, dtype=float)
    n = len(profile)

    for _ in range(passes):
        for i in range(n - 1, -1, -1):
            next_i = (i + 1) % n
            reachable_speed = np.sqrt(
                profile[next_i] ** 2
                + 2 * braking * segment_lengths[i]
            )
            profile[i] = min(profile[i], reachable_speed)

    return profile

def speed_controller(
    target_speed: float,
    speed: float,
    gain: float = 1.5,
    curvature: float = 0.0,
    total_limit: float = 11.0,
) -> float:
    """Return a feasible acceleration command.

    Args:
        target_speed: Desired speed in m/s.
        speed: Current measured speed in m/s.
        gain: Proportional speed gain in 1/s.
        curvature: Current/reference curvature in 1/m.
        total_limit: Conservative acceleration-circle radius in m/s^2.

    Returns:
        Scalar longitudinal acceleration command in m/s^2.

    Proportional feedback converts speed error into requested acceleration.
    Constrain it by both the curvature-dependent acceleration budget and the
    actuator's asymmetric range [-10, 4].
    """
    difference = target_speed - speed
    a= gain * difference
    budget = longitudinal_budget(speed,curvature,total_limit,)
    safe_a = np.clip(a, -budget, budget,)
    return np.clip(safe_a, -10, 4)


def run_checks():
    print("Lesson 5 checks")
    check_close("curvature_speed_limit(): straight uses speed cap", curvature_speed_limit(0), 20)
    check_close("curvature_speed_limit(): kappa=0.1 gives 10 m/s", curvature_speed_limit(.1), 10)
    check_close("curvature_speed_limit(): accepts arrays element-wise", curvature_speed_limit(np.array([0, .1, .4])), [20, 10, 5])
    check_close("longitudinal_budget(): straight preserves full budget", longitudinal_budget(10, 0), 11)
    check_close("longitudinal_budget(): lateral demand consumes budget", longitudinal_budget(10, .08), np.sqrt(121-64))
    raw = np.array([20., 20., 5., 20.])
    profile = braking_pass(raw, np.ones(4), braking=5, passes=4)
    check_true("braking_pass(): low corner speed propagates backward", profile[1] < 6 and profile[0] < 7, f"Expected profile[1] < 6 and profile[0] < 7; got {profile}.")
    check_close("speed_controller(): acceleration clips at actuator maximum", speed_controller(10, 0), 4)
    check_true("speed_controller(): command fits remaining acceleration circle", abs(speed_controller(20, 10, curvature=.1)) <= np.sqrt(21) + 1e-9)
    finish(5, f"your planner slows from {raw[0]:.1f} to {profile[0]:.2f} m/s before the corner")


if __name__ == "__main__":
    run_checks()
