import numpy as np
from simulator import Simulator, centerline

#Constant values
TRACK_LENGTH = 104.75797893910486

WHEELBASE = 1.58

STEERING_MIN = -0.7
STEERING_MAX = 0.7

STEERING_RATE_MIN = -1.0
STEERING_RATE_MAX = 1.0

ACCELERATION_MIN = -10.0
ACCELERATION_MAX = 4.0
TOTAL_ACCELERATION_LIMIT = 12.0


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float)


def global_to_local(car_position, car_heading, point):
    return rotation(-car_heading) @ (np.asarray(point, dtype = float) - np.asarray(car_position, dtype = float))


def nearest_path_index(position, path, previous_index = None, search_window = 40):
    position = np.asarray(position, dtype = float)
    path = np.asarray(path, dtype = float)
    offsets = path - position
    distances = np.linalg.norm(offsets, axis = 1)
    return np.argmin(distances)


def path_headings(path):
    path = np.asarray(path, dtype = float)
    next = np.roll(path, -1, axis=0)
    previous = np.roll(path, 1, axis=0)
    displacements = next - previous
    headings = np.arctan2(displacements[:, 1], displacements[:, 0])
    return headings


def path_curvature(path):
    headings = path_headings(path)
    next = np.roll(headings, -1, axis=0)
    previous = np.roll(headings, 1, axis=0)
    change = (next - previous + np.pi) % (2 * np.pi) - np.pi
    path = np.asarray(path, dtype = float)
    next_path = np.roll(path, -1, axis = 0)
    previous_path = np.roll(path, 1, axis=0)
    distance = np.linalg.norm(next_path - previous_path, axis=1)
    return change / distance


def advance_index(index, distance, path):
    current = index
    path = np.asarray(path, dtype = float)
    total_distance = 0

    while total_distance < distance:
        previous_point = path[current]
        current = (current + 1) % len(path)
        current_point = path[current]
        total_distance += np.linalg.norm(current_point - previous_point)

    return current


def pure_pursuit_steering(local_target, wheelbase = WHEELBASE):
    x_L = local_target[0]
    y_L = local_target[1]
    distance = np.sqrt(x_L ** 2 + y_L ** 2)
    k = 2 * y_L / (distance ** 2)
    steering = np.atan(wheelbase * k)
    return steering


def speed_dependent_lookahead(speed, base = 2.0, gain = 0.35, maximum = 8.0):
    return np.clip(base + gain * abs(speed), base, maximum)


def steering_rate_servo(desired_steering, current_steering, gain = 4.0):
    return np.clip((desired_steering - current_steering) * gain, -1, 1)


def curvature_speed_limit(curvature, lateral_limit = 10.0, straight_speed = 20.0, epsilon = 1e-6):
    curvature = np.abs(np.asarray(curvature, dtype=float))
    safe_curvature = np.maximum(curvature, epsilon)
    corner_speed = np.sqrt(lateral_limit / safe_curvature)
    return np.minimum(corner_speed, straight_speed)


def longitudinal_budget(speed, curvature, total_limit = 11.0):
    lateral = speed**2 * abs(curvature)
    remaining_squared = np.maximum(total_limit**2 - lateral**2, 0.0)
    return float(np.sqrt(remaining_squared))


def braking_pass(raw_speed_limits, segment_lengths, braking = 8.0, passes = 2):
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


def speed_controller(target_speed, speed, gain = 1.5, curvature = 0.0, total_limit = 11.0):
    difference = target_speed - speed
    a= gain * difference
    budget = longitudinal_budget(speed,curvature,total_limit,)
    safe_a = np.clip(a, -budget, budget,)
    return np.clip(safe_a, -10, 4)


class ProjectController:
    def __init__(self, centerline, sample_count = 1200):
        self.s_values = np.linspace(0, TRACK_LENGTH, sample_count, endpoint=False)
        self.path = centerline(self.s_values)
        self.curvature = path_curvature(self.path)
        self.previous_index = None
        self.speed_profile = self.make_speed_profile()

    def make_speed_profile(self):
        path = np.asarray(self.path, dtype = float)
        curvature = path_curvature(path)
        curve_limit = curvature_speed_limit(curvature, lateral_limit=8, straight_speed=12)
        next = np.roll(path, -1, axis=0)
        displacements = next - path
        lengths = np.linalg.norm(displacements, axis=1)
        profile = braking_pass(curve_limit, lengths, braking=7)
        return profile

    def __call__(self, state):
        state = np.asarray(state, dtype = float)
        position = state[0:2]
        heading = state[2]
        speed = state[3]
        steering = state[4]
        speed_profile = np.asarray(self.speed_profile, dtype = float)
        path = np.asarray(self.path, dtype = float)

        current_index = nearest_path_index(position, path, self.previous_index)
        lookahead = speed_dependent_lookahead(speed)
        target_index = advance_index(current_index, lookahead, path)
        target = path[target_index]
        target = global_to_local(position, heading, target)
        steer = steering_rate_servo(np.clip(pure_pursuit_steering(target), -0.7, 0.7), steering)

        target_speed = speed_profile[current_index]
        curvatures = path_curvature(path)
        current_k = self.curvature[current_index]

        acceleration = speed_controller(target_speed, speed, curvature=current_k)
        self.previous_index = current_index
        return np.array([acceleration, steer], dtype = float)

first_iteration = ProjectController(centerline)


sim = Simulator()



def controller(x):
    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]
    """
    # xpos   = x[0]                   # current x position
    # ypos   = x[1]                   # current y position
    # phi    = np.mod(x[2], 2*np.pi)  # current heading (radians)
    # v      = x[3]                   # current velocity
    # theta   = x[4]                  # current steering angle

    return first_iteration(x)




sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()
