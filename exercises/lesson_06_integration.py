"""Lesson 6: assemble a closed-loop controller and watch it drive.

Run normally to see a plot. `--no-plot` runs only the automated simulation.

This lesson mostly requires composition rather than new mathematics. Values
returned by earlier helpers must remain aligned: `path[i]`, `curvature[i]`, and
`speed_profile[i]` all describe the same path location.

Relevant syntax:

    state = np.asarray(state, dtype=float)
    position = state[:2]
    heading, speed, steering = state[2], state[3], state[4]

    result = helper(argument, named_option=value)
    control = np.array([acceleration, steering_rate], dtype=float)
    control, current_index, target_index = controller_command(...)

Imported helpers are ordinary functions. A tuple return can be unpacked into
multiple names, and a returned index can be passed back as `previous_index`
on the next call. Use intermediate names that retain physical meaning—such as
`local_target`, `desired_steering`, and `target_speed`—so angle, rate, index,
distance, and speed are not accidentally interchanged.

When debugging the integrated controller, inspect the shape and meaning of
intermediate values at the branch where behaviour first becomes implausible.
The lateral branch should end in steering rate; the longitudinal branch should
end in acceleration; the final control must have shape `(2,)` in that order.
"""

import sys
import numpy as np
from numpy.typing import ArrayLike, NDArray

from course_utils import check_true, circle_path, finish
from lesson_01_geometry import global_to_local
from lesson_02_dynamics import net_acceleration, rk4_step
from lesson_03_path import advance_index, nearest_path_index, path_curvature
from lesson_04_pure_pursuit import (
    pure_pursuit_steering,
    speed_dependent_lookahead,
    steering_rate_servo,
)
from lesson_05_speed_planning import (
    braking_pass,
    curvature_speed_limit,
    speed_controller,
)


def build_speed_profile(path: ArrayLike) -> NDArray[np.float64]:
    """Return one target speed per path point.

    Args:
        path: ndarray with shape (N, 2), ordered around a closed loop.

    Returns:
        ndarray with shape (N,), one positive target speed in m/s per point.

    Curvature provides a local speed limit, while wrapped segment lengths let
    the braking pass propagate future corner constraints backward. The result
    stays aligned one-to-one with the path. Use the conservative course values
    lateral_limit=8, straight_speed=12, and braking=7.
    """
    path = np.asarray(path, dtype = float)
    curvature = path_curvature(path)
    curve_limit = curvature_speed_limit(curvature, lateral_limit=8, straight_speed=12)
    next = np.roll(path, -1, axis=0)
    displacements = next - path
    lengths = np.linalg.norm(displacements, axis=1)
    profile = braking_pass(curve_limit, lengths, braking=7)
    return profile



def controller_command(
    state: ArrayLike,
    path: ArrayLike,
    speed_profile: ArrayLike,
    previous_index: int | None,
) -> tuple[NDArray[np.float64], int, int]:
    """Return (control, nearest_index, target_index).

    Args:
        state: ndarray shape (5,), ordered
            [x, y, heading, speed, steering_angle].
        path: ndarray shape (N, 2), closed and ordered.
        speed_profile: ndarray shape (N,), aligned one-to-one with path.
        previous_index: Last nearest index, or None on the first call.

    Returns:
        Tuple (control, nearest_index, target_index), where control is a
        shape-(2,) ndarray [acceleration, steering_rate] and both indices are
        integers in [0, N).

    Localization supplies a shared current index. The lateral branch selects a
    speed-dependent target, transforms it into the car frame, and converts
    pure-pursuit steering into a bounded steering rate. The longitudinal branch
    compares current speed with the aligned speed profile while accounting for
    local curvature. Return the actuator command plus both indices so the
    simulation can preserve progress and expose its chosen target.
    """
    state = np.asarray(state, dtype = float)
    position = state[0:2]
    heading = state[2]
    speed = state[3]
    steering = state[4]
    speed_profile = np.asarray(speed_profile, dtype = float)
    path = np.asarray(path, dtype = float)

    current_index = nearest_path_index(position, path, previous_index)
    lookahead = speed_dependent_lookahead(speed)
    target_index = advance_index(current_index, lookahead, path)
    target = path[target_index]
    target = global_to_local(position, heading, target)
    steer = steering_rate_servo(np.clip(pure_pursuit_steering(target), -0.7, 0.7), steering)

    target_speed = speed_profile[current_index]
    curvatures = path_curvature(path)
    current_k = curvatures[current_index]

    acceleration = speed_controller(target_speed, speed, current_k)
    return np.array([acceleration, steer], dtype = float), current_index, target_index



def simulate(path, speed_profile, duration=14.0, dt=.01):
    """Run the supplied miniature simulation. You do not need to edit this."""
    radius_guess = np.mean(np.linalg.norm(path, axis=1))
    state = np.array([radius_guess, 0.0, np.pi/2, 0.0, 0.0])
    previous_index = None
    states, controls, accelerations, targets = [], [], [], []
    for _ in range(int(duration / dt)):
        control, previous_index, target = controller_command(
            state, path, speed_profile, previous_index
        )
        control = np.asarray(control, dtype=float)
        if control.shape != (2,):
            raise AssertionError(f"controller must return control shape (2,), got {control.shape}")
        states.append(state.copy())
        controls.append(control.copy())
        accelerations.append(net_acceleration(state, control))
        targets.append(target)
        state = rk4_step(state, control, dt)
    return map(np.asarray, (states, controls, accelerations, targets))


def run_checks(show_plot=True):
    print("Lesson 6 checks")
    path = circle_path(radius=12, count=600)
    speeds = build_speed_profile(path)
    check_true("build_speed_profile(): returns one value per path point", np.asarray(speeds).shape == (len(path),), f"Expected shape ({len(path)},); got {np.asarray(speeds).shape}.")
    check_true("build_speed_profile(): all speeds are positive and finite", np.all(np.isfinite(speeds)) and np.min(speeds) > 0)
    states, controls, accelerations, targets = simulate(path, speeds)
    radial_error = np.abs(np.linalg.norm(states[:, :2], axis=1) - 12)
    angular_progress = np.unwrap(np.arctan2(states[:, 1], states[:, 0]))
    check_true("controller_command(): car makes substantial forward progress", angular_progress[-1] > 4.0, f"Expected > 4.0 rad; got {angular_progress[-1]:.2f} rad. Check acceleration and steering direction.")
    check_true("controller_command(): car remains within 2 m of path", np.max(radial_error) < 2.0, f"Maximum radial error was {np.max(radial_error):.2f} m. Check local frame, steering sign, gain, and lookahead.")
    check_true("controller_command(): net acceleration remains <= 12", np.max(accelerations) <= 12.0, f"Maximum was {np.max(accelerations):.2f} m/s^2. Lower speed/acceleration limits.")
    check_true("controller_command(): both control components respect bounds", np.all((controls[:, 0] >= -10) & (controls[:, 0] <= 4)) and np.all(np.abs(controls[:, 1]) <= 1), "Expected acceleration in [-10, 4] and steering rate in [-1, 1].")
    finish(6, f"your car traveled {angular_progress[-1]:.2f} radians with max path error {np.max(radial_error):.2f} m")

    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.plot(path[:, 0], path[:, 1], "k--", label="reference path")
        plt.plot(states[:, 0], states[:, 1], color="tab:blue", label="your car")
        plt.scatter(states[0, 0], states[0, 1], color="tab:green", label="start")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.title("Lesson 6: closed-loop path tracking")
        plt.show()


if __name__ == "__main__":
    run_checks(show_plot="--no-plot" not in sys.argv)
