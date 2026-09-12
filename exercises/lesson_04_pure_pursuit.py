"""Lesson 4: turn path geometry into steering commands.

Pure pursuit expects a target already expressed as `[forward, left]` in the
car frame. It produces a desired steering angle; a separate feedback servo
turns that desired angle into the steering-rate command accepted by the model.

Relevant syntax:

    target = np.asarray(local_target, dtype=float)
    forward, left = target
    distance_squared = forward**2 + left**2
    angle = np.arctan(wheelbase * curvature)
    bounded = np.clip(value, lower, upper)
    lookahead = base + gain * abs(speed)

`np.clip` works on a scalar or element-wise on an array. A near-zero target
distance needs a guard because the pure-pursuit curvature divides by squared
distance. Keep the two control quantities distinct while working: desired
steering is measured in radians, while steering rate is measured in rad/s.

Sanity checks help with signs: a target directly ahead should request zero
steering; a target ahead-left should request positive steering.
"""

import numpy as np
from numpy.typing import ArrayLike
from course_utils import check_close, check_true, finish

WHEELBASE = 1.58


def pure_pursuit_steering(local_target: ArrayLike, wheelbase: float = WHEELBASE) -> float:
    """Return desired steering angle for a target expressed in car coordinates.

    Args:
        local_target: Array-like [forward, left] with shape (2,), in metres.
        wheelbase: Positive axle-to-axle distance in metres.

    Returns:
        Scalar desired steering angle in radians; positive means left.

    Pure pursuit fits a turning arc toward the local target. Its curvature is
    2*left/distance^2, which the bicycle geometry converts to steering with
    atan(wheelbase*curvature). A target at zero distance has no direction.
    """
    x_L = local_target[0]
    y_L = local_target[1]
    distance = np.sqrt(x_L ** 2 + y_L ** 2)
    k = 2 * y_L / (distance ** 2)
    steering = np.atan(wheelbase * k)
    return steering


def speed_dependent_lookahead(
    speed: float,
    base: float = 2.0,
    gain: float = 0.35,
    maximum: float = 8.0,
) -> float:
    """Choose how far ahead the path tracker should aim.

    Args:
        speed: Current signed vehicle speed in m/s.
        base: Minimum lookahead in metres.
        gain: Extra metres of lookahead per m/s of speed.
        maximum: Maximum lookahead in metres.

    Returns:
        Scalar lookahead distance in [base, maximum].

    Faster motion needs a more distant target for smoother, more predictive
    steering. Use speed magnitude and keep the distance within its bounds.
    """
    return np.clip(base + gain * abs(speed), base, maximum)


def steering_rate_servo(
    desired_steering: float,
    current_steering: float,
    gain: float = 4.0,
) -> float:
    """Convert desired steering angle into a feasible actuator command.

    Args:
        desired_steering: Target front-wheel angle in radians.
        current_steering: Measured front-wheel angle in radians.
        gain: Proportional feedback gain in 1/s.

    Returns:
        Scalar steering rate in rad/s, clipped to [-1, 1].

    Proportional feedback turns steering-angle error into steering rate; the
    actuator bound prevents an infeasible command.
    """
    return np.clip((desired_steering - current_steering) * gain, -1, 1)


def run_checks():
    print("Lesson 4 checks")
    check_close("pure_pursuit_steering(): target straight ahead gives zero", pure_pursuit_steering([5, 0]), 0)
    check_true("pure_pursuit_steering(): left target gives positive steering", pure_pursuit_steering([5, 2]) > 0)
    check_close("pure_pursuit_steering(): applies 2*y/Ld^2 geometry", pure_pursuit_steering([3, 4]), np.arctan(WHEELBASE * 8/25))
    check_close("speed_dependent_lookahead(): stopped car uses base", speed_dependent_lookahead(0), 2)
    check_close("speed_dependent_lookahead(): grows with speed magnitude", speed_dependent_lookahead(10), 5.5)
    check_close("speed_dependent_lookahead(): respects maximum", speed_dependent_lookahead(100), 8)
    check_close("steering_rate_servo(): proportional unsaturated case", steering_rate_servo(.2, .1), .4)
    check_close("steering_rate_servo(): clips rate at +1 rad/s", steering_rate_servo(.7, -.7), 1)
    finish(4, "your geometry now produces feasible steering-rate commands")


if __name__ == "__main__":
    run_checks()
