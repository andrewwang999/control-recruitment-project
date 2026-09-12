"""Lesson 2: bicycle dynamics and RK4 integration.

A dynamics function returns rates of change, not the next state. Its output has
the same five positions as the state, but the units are per second. An
integrator uses those rates to estimate a later state.

Relevant syntax:

    state = np.asarray(state, dtype=float)
    x, y, heading, speed, steering = state
    acceleration, steering_rate = control
    beta = np.arctan(0.5 * np.tan(steering))
    signed_drag = coefficient * np.sign(speed) * speed**2
    derivative = np.array([rate_0, rate_1, rate_2, rate_3, rate_4], dtype=float)

Arithmetic between same-shaped ndarrays is element-wise, so an intermediate
integration state has the general form:

    trial_state = state + time_fraction * derivative

RK4 evaluates the derivative at several such trial states. Every slope and
state remains shape `(5,)`; only their physical interpretation and units
differ. Check limiting cases while working: a stopped car with zero control
should have zero derivative, and zero steering should produce no lateral or
heading rate.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from course_utils import check_close, check_true, finish

DT = 0.01
HALF_WHEELBASE = 0.79
DRAG = 0.006133333333333333


def bicycle_derivative(state: ArrayLike, control: ArrayLike) -> NDArray[np.float64]:
    """Return d/dt [x, y, heading, speed, steering].

    Args:
        state: Array-like shape (5,), ordered
            [x, y, heading, speed, steering_angle].
        control: Array-like shape (2,), ordered
            [longitudinal_acceleration, steering_rate].

    Returns:
        Floating-point ndarray with shape (5,), ordered
        [x_rate, y_rate, heading_rate, speed_rate, steering_rate].

    The kinematic bicycle model converts speed and steering into planar-motion
    rates. Longitudinal acceleration changes speed after drag, while commanded
    steering rate directly changes steering angle. Use
    beta = atan(0.5*tan(steering)); quadratic drag opposes motion.
    """
    beta = np.arctan(0.5 * np.tan(state[4]))
    x_rate = np.cos(state[2] + beta) * state[3] 
    y_rate = np.sin(state[2] + beta) * state[3]
    heading_rate = (state[3] / HALF_WHEELBASE) * np.sin(beta)
    speed_rate = control[0] - DRAG * state[3] ** 2
    steering_rate = control[1]

    return np.array([x_rate, y_rate, heading_rate, speed_rate, steering_rate], dtype = float)


def rk4_step(state: ArrayLike, control: ArrayLike, dt: float = DT) -> NDArray[np.float64]:
    """Advance one state by one timestep using classical RK4.

    Args:
        state: Array-like current state with shape (5,).
        control: Array-like constant control over this step, shape (2,).
        dt: Positive timestep in seconds.

    Returns:
        ndarray next state with shape (5,).

    RK4 estimates the derivative at the start, two midpoint states, and an end
    state, then combines those slopes with weights 1, 2, 2, 1.
    """
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    k1 = bicycle_derivative(state, control)
    k2 = bicycle_derivative(state + k1 * dt/2, control)
    k3 = bicycle_derivative(state + k2 * dt/2, control)
    k4 = bicycle_derivative(state + dt * k3, control)

    state_next = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state_next


def net_acceleration(state: ArrayLike, control: ArrayLike) -> float:
    """Compute the simulator's combined acceleration magnitude.

    Args:
        state: Array-like shape (5,); only speed and steering are needed.
        control: Array-like shape (2,); only longitudinal acceleration is used.

    Returns:
        Nonnegative scalar acceleration magnitude in m/s^2.

    Turning produces lateral acceleration while the control supplies
    longitudinal acceleration. Their perpendicular components combine as a
    Euclidean magnitude.
    """
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    beta = np.arctan(0.5 * np.tan(state[4]))
    a_lat = state[3] ** 2 / 0.79 * np.sin(beta)
    a_lon = control[0]
    return np.sqrt(a_lat ** 2 + a_lon ** 2)


def run_checks():
    print("Lesson 2 checks")
    zero = np.zeros(5)
    check_close("bicycle_derivative(): stationary car has zero derivative", bicycle_derivative(zero, [0, 0]), np.zeros(5))
    check_close("bicycle_derivative(): straight-line rates and drag", bicycle_derivative([0, 0, 0, 10, 0], [2, .4]), [10, 0, 0, 2-DRAG*100, .4])
    turning = bicycle_derivative(np.array([0, 0, 0, 8, .2]), np.array([0, 0]))
    check_true("bicycle_derivative(): positive steering turns left", turning[1] > 0 and turning[2] > 0)
    one_second = np.zeros(5)
    for _ in range(100):
        one_second = rk4_step(one_second, np.array([1.0, 0.0]))
    check_true("rk4_step(): repeated integration moves and accelerates", one_second[0] > 0.45 and one_second[3] > 0.9)
    check_close("net_acceleration(): straight stationary case", net_acceleration(zero, [4, 0]), 4)
    check_true("net_acceleration(): fast turn exceeds 12 m/s^2", net_acceleration([0, 0, 0, 12, .3], [0, 0]) > 12)
    finish(2, f"your numerical car moved {one_second[0]:.3f} m in its first second")


if __name__ == "__main__":
    run_checks()
