# 🏎️ FEB Controls Crash Course: Lessons 0–6

This is the companion guide for the exercise course in this folder. Start with
the NumPy lesson and work forward through the integrated controller in Lesson
6. Each lesson introduces one layer of the final system, then gives you small
functions through which to practise that idea.

The guide explains the concepts, the mathematics, the meanings of the inputs
and outputs, and the general shape an implementation should have. The lesson
files contain the exact function signatures, type annotations, checks, and
local hints. You should use the two side by side rather than treating this file
as a replacement for the exercises.

## What the lessons cover

| Lesson | Exercise file | What you learn | What it contributes |
|---:|---|---|---|
| 0 | `lesson_00_numpy.py` | Arrays, shapes, slicing, broadcasting, norms, clipping, and rolling | The numerical vocabulary used everywhere else |
| 1 | `lesson_01_geometry.py` | Rotations, coordinate frames, and angle wrapping | Expressing a global target relative to the car |
| 2 | `lesson_02_dynamics.py` | State derivatives, the bicycle model, drag, Euler, and RK4 | Predicting how commands change vehicle motion |
| 3 | `lesson_03_path.py` | Nearest-point search, path headings, curvature, and distance-based index advancement | Understanding a sampled closed centerline |
| 4 | `lesson_04_pure_pursuit.py` | Lookahead targeting, pure-pursuit geometry, and steering feedback | Producing a stable lateral-control command |
| 5 | `lesson_05_speed_planning.py` | Curvature speed limits, braking feasibility, and the shared acceleration budget | Producing safe target speeds and acceleration |
| 6 | `lesson_06_integration.py` | Connecting localization, steering, and speed control | Running the complete learned controller on the practice track |

The lessons build toward this pipeline:

```text
NumPy representation
→ coordinate transforms and vehicle motion
→ sampled path geometry
→ pure-pursuit steering
→ curvature-aware speed planning
→ integrated controller
```

## How to use this guide

Keep the current lesson open beside the matching section of this guide. Before
implementing a function, make sure you can state what each parameter means,
its expected shape and units, and what the returned value represents. Then use
the “Implementation shape” blocks to translate the idea into your own Python.

Run the current lesson after each small piece of work. Its checks are intended
to tell you which function failed and whether the problem is a value, type,
shape, sign, unit, or boundary error. Fix the first failed check before relying
on later results, because later functions often build on earlier ones.

From the repository root, an individual lesson can be run with, for example:

```bash
python3 exercises/lesson_02_dynamics.py
```

The whole course can be checked with:

```bash
python3 exercises/check_all.py
```

## Contents

| Lesson | Central question |
|---|---|
| [0. NumPy and numerical representation](#lesson-0--numpy-and-numerical-representation) | How is vehicle and track data represented? |
| [1. Coordinate frames and angles](#lesson-1--coordinate-frames-and-angles) | Relative to what is a direction measured? |
| [2. Vehicle dynamics](#lesson-2--vehicle-dynamics) | How do steering and acceleration change motion? |
| [3. Sampled path geometry](#lesson-3--sampled-path-geometry) | Where is the car and how does the road bend? |
| [4. Pure-pursuit lateral control](#lesson-4--pure-pursuit-lateral-control) | How does the car steer toward the path? |
| [5. Speed planning](#lesson-5--speed-planning-and-the-acceleration-limit) | How fast can the car safely travel? |
| [6. Controller integration](#lesson-6--complete-controller-integration) | How do all the learned components operate together? |

---

## Lesson 0 — NumPy and numerical representation

### State and control

The simulator represents the car with a five-element state:

```text
state = [x, y, heading, speed, steering_angle]
```

The controller returns a two-element command:

```text
control = [longitudinal_acceleration, steering_rate]
```

This distinction is fundamental. Steering angle is something the vehicle
currently has; steering rate is something the controller directly requests.
The controller cannot instantaneously place the wheels at an angle. It commands
them to move toward that angle over time.

### Array shapes carry meaning

The main shapes used in the project are:

| Shape | Meaning |
|---|---|
| `(5,)` | One vehicle state |
| `(2,)` | One control or one 2D point |
| `(N, 2)` | `N` points, one point per row |
| `(N,)` | One scalar value per path point |
| `(5, N)` | Simulator state history, one state variable per row |

Shape is not merely a storage detail. It determines which operations represent
the intended mathematics. A collection of points shaped `(N, 2)` supports one
row-wise distance per point. A transposed `(2, N)` array represents the same
numbers but requires different operations.

### Vectorization and broadcasting

NumPy can compare one position to every path point without an explicit Python
loop:

```text
path shape:      (N, 2)
position shape:     (2,)
path - position: (N, 2)
```

The position is broadcast across every row. Taking a norm with `axis=1`
produces one distance per point:

```text
offsets:   (N, 2)
distances: (N,)
```

This pattern appears in localization, tracking-error evaluation, and collision
geometry.

#### Implementation shape

Nearest-point calculations generally have this structure:

```text
one position + many points
→ subtract the position from every row
→ take one norm per row
→ select the smallest distance or its index
```

The key decision is the reduction axis. With points stored as `(N, 2)`, each
row contains one vector, so row-wise norms use `axis=1` and produce `(N,)`.

### Indexing, slicing, and transpose

The meaning of the state is fixed by index:

```text
state[:2] → position
state[2]  → heading
state[3]  → speed
state[4]  → steering angle
```

The simulator history is shaped `(5, N)`, so extracting positions requires the
first two rows and a transpose:

```text
states[:2].T → shape (N, 2)
```

### Copying versus sharing data

Converting an existing NumPy array with `np.asarray` may return a view of the
same data. Modifying it may modify the caller's array. When a function should
return a changed version while preserving its input, an explicit copy matters.

This arose when constraining controls: clipping should produce a safe command
without unexpectedly mutating the original object.

The common implementation pattern is:

```text
array-like input
→ convert to a floating NumPy array
→ copy if the result will be modified
→ perform vector or component operations
→ return a value with the documented shape
```

---

## Lesson 1 — Coordinate frames and angles

### Global frame versus car frame

Track points are stored in a fixed global coordinate frame. The controller,
however, needs answers relative to the moving car:

```text
Is the target ahead or behind?
Is it left or right?
```

A global displacement is:

```text
d_global = target_position − car_position
```

It becomes a car-relative displacement by applying the inverse of the car's
heading rotation:

```text
d_local = R(−heading) d_global
```

The resulting coordinates mean:

```text
d_local[0] > 0  → ahead
d_local[0] < 0  → behind
d_local[1] > 0  → left
d_local[1] < 0  → right
```

### Rotation matrix

Positive angles rotate counterclockwise:

```text
       ┌ cos α   −sin α ┐
R(α) = │                │
       └ sin α    cos α ┘
```

The inverse rotation is:

```text
R⁻¹(α) = R(−α) = Rᵀ(α)
```

Using `−heading` for global-to-local conversion was not an arbitrary sign
choice. It removes the car's global orientation so the car becomes the local
reference direction.

#### Implementation shape

The conversion has a small, reusable data flow:

```text
car position (2,) + target position (2,) + heading scalar
→ global displacement (2,)
→ inverse-heading rotation (2×2)
→ local displacement [forward, left] (2,)
```

The key idea is to subtract positions before rotating. The positions create a
displacement vector; the heading determines the coordinate basis in which that
vector is described.

### Equivalent angles and wrapping

Angles differing by a whole rotation represent the same direction:

```text
0, 2π, and −2π are equivalent
```

Naive angle subtraction fails near the `−π/π` boundary. A heading of `179°`
and a target of `−179°` differ physically by only `2°`, not `358°`.

Wrapping maps an angle to the standard interval `[-π, π)`:

```text
wrap(α) = (α + π) mod 2π − π
```

This is important for heading errors and numerical curvature.

---

## Lesson 2 — Vehicle dynamics

### A model describes rates of change

The dynamic model does not directly answer where the car will be far in the
future. It answers how each state variable is changing now:

```text
state derivative = [x_rate, y_rate, heading_rate, speed_rate, steering_rate]
```

| Quantity | Unit | Rate unit |
|---|---:|---:|
| x, y position | m | m/s |
| heading | rad | rad/s |
| speed | m/s | m/s² |
| steering angle | rad | rad/s |

Over a small interval, change is approximately rate multiplied by time:

```text
x_rate = 10 m/s
dt = 0.01 s
change in x ≈ 0.10 m
```

### Kinematic bicycle model

The four physical wheels are simplified into one front and one rear wheel. A
small body-velocity angle accounts for the center of the car moving in a
slightly different direction from the body heading:

```text
β = atan(0.5 tan θ)
```

The motion rates are approximately:

```text
ẋ = v cos(φ + β)
ẏ = v sin(φ + β)
φ̇ = (v / 0.79) sin β
```

Here:

```text
φ = vehicle heading
θ = steering angle
v = forward speed
```

Positive steering creates positive curvature and a counterclockwise left turn.
At zero speed, steering alone cannot rotate the car in place.

### Speed dynamics and drag

Requested acceleration is reduced by signed quadratic drag:

```text
v̇ = acceleration − drag_coefficient × sign(v) × v²
```

The `v²` term means drag grows rapidly at higher speed. A constant positive
acceleration therefore does not produce unlimited speed.

### Numerical integration

The derivative must be integrated to advance the state. Euler integration uses
only the beginning derivative:

```text
next state = state + dt × derivative
```

RK4 samples four derivatives—beginning, two midpoint estimates, and an end
estimate—and combines them:

```text
k₁ = f(q, u)
k₂ = f(q + dt k₁/2, u)
k₃ = f(q + dt k₂/2, u)
k₄ = f(q + dt k₃,   u)

q_next = q + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

This matters because the derivative itself changes during a turn as heading,
speed, and steering change.

The final controller does not need to implement its own dynamics; the supplied
simulator performs integration. Understanding the model is still necessary for
choosing safe and effective commands.

#### Implementation shape

The dynamics function maps one state and one command to five instantaneous
rates:

```text
state (5,) + control (2,)
→ unpack the physical quantities
→ calculate β and the five rate equations
→ state derivative (5,)
```

An integrator then combines derivative evaluations with the current state:

```text
current state + derivative function + dt
→ one or more slope estimates
→ next state with the same shape
```

Keeping these jobs separate is useful: the model describes the physics, while
the integrator describes how a computer approximates motion through time.

---

## Lesson 3 — Sampled path geometry

### A path as ordered points

The continuous centerline is sampled into an ordered array:

```text
point 0 → point 1 → ... → point N−1 → point 0
```

The ordering defines forward progress. Because the path is closed, the final
point connects to the first.

Modulo indexing expresses that wraparound:

```text
(N−1 + 1) mod N = 0
```

### Nearest-point localization

The controller estimates its track location by finding the centerline point
nearest to the car's global position:

```text
current_index = argmin distance(car_position, path[i])
```

The previous index provides continuity. A car moves only a short distance in
one 0.01-second timestep, so the new nearest index should normally be close to
the preceding one.

A local wrapped search is safer than a global search on tracks where separate
sections pass close together. Without it, localization could jump across a
hairpin to the wrong piece of track.

The nearest index is not assumed to be `previous_index + 1`. The vehicle may:

- Remain closest to the same sample for several calls.
- Advance across multiple samples in one call.
- Wrap from the last sample to the first.

### Tangent heading

A point alone has no direction. Neighboring points define the local path
tangent:

```text
tangent[i] ≈ path[i+1] − path[i−1]
```

The tangent vector becomes a heading through `atan2(y, x)`. Centered
differences reduce directional bias compared with using only the next segment.

### Curvature

Curvature measures how rapidly path heading changes with distance:

```text
κ = d(path heading) / d(distance)
```

Numerically:

```text
     wrap(heading[i+1] − heading[i−1])
κ ≈ ──────────────────────────────────
          ‖path[i+1] − path[i−1]‖
```

For a circle:

```text
κ = 1 / radius
```

Examples:

| Radius | Curvature | Interpretation |
|---:|---:|---|
| 100 m | `0.01 m⁻¹` | Gentle turn |
| 10 m | `0.10 m⁻¹` | Tighter turn |
| 2 m | `0.50 m⁻¹` | Very tight turn |
| Infinite | `0` | Straight |

Curvature sign describes turn direction. Curvature magnitude describes
sharpness and is what the speed planner mainly uses.

### Physical lookahead distance

Path samples are not guaranteed to be perfectly uniform. Therefore a target is
chosen by accumulating physical segment lengths rather than adding a fixed
number of indices.

This separates two quantities that are easy to confuse:

```text
lookahead distance → metres
target index       → row number in the sampled path
```

#### Implementation shapes

The path operations are variations on comparing aligned rows, then reducing or
accumulating the result:

```text
car position + candidate path rows
→ one distance per candidate
→ smallest distance and its path index

path rows
→ previous and next rows with closed-loop wrapping
→ tangent vector and heading per row

heading array + path rows
→ wrapped heading change / physical distance
→ curvature array aligned with the path

starting index + lookahead distance + path
→ accumulate successive segment lengths
→ first index that reaches the lookahead distance
```

For a closed path, wrapping is part of the geometry rather than a special
failure case. Every use of “next” or “previous” must still make sense at
indices `0` and `N−1`.

---

## Lesson 4 — Pure-pursuit lateral control

### Chasing a point ahead

Pure pursuit repeatedly selects a centerline point ahead of the car and
calculates a circular arc toward it. The target is first expressed in the car
frame:

```text
local target = [forward displacement, left displacement]
```

Examples:

```text
[5, 0]   → five metres directly ahead
[5, 2]   → five metres ahead and two metres left
[5, −2]  → five metres ahead and two metres right
```

### Pure-pursuit geometry

For local target `(x_L, y_L)`:

```text
L_d² = x_L² + y_L²
κ_command = 2 y_L / L_d²
desired steering = atan(wheelbase × κ_command)
```

The lateral target coordinate controls steering sign. The full target distance
controls correction strength. A distant target with the same lateral offset
produces a gentler command because the vehicle has more room to correct.

### Lookahead tradeoff

| Lookahead | Benefit | Cost |
|---|---|---|
| Short | Tight, quick path correction | Oscillation and abrupt steering |
| Long | Smooth, stable steering | Corner cutting and delayed correction |

Lookahead grows with speed because a faster vehicle consumes the same physical
distance in less time:

```text
lookahead = base + gain × |speed|
```

### Desired steering versus steering-rate command

Pure pursuit produces the steering angle that would geometrically point the car
toward the target. The actuator accepts a rate, so another feedback loop is
needed:

```text
steering error = desired angle − current angle
steering rate  = gain × steering error
```

The desired angle is constrained to `[-0.7, 0.7] rad`; steering rate is
constrained to `[-1, 1] rad/s`.

This is a nested control structure:

```text
path error → desired steering angle → steering error → steering-rate command
```

#### Implementation shape

The lateral controller turns geometry into an actuator command in two distinct
stages:

```text
local target [forward, left]
→ target distance and geometric curvature
→ desired steering angle

desired steering angle + measured steering angle
→ steering error
→ bounded steering-rate command
```

Preserving this distinction prevents a common conceptual error: pure pursuit
produces an angle, while the simulator accepts the rate at which that angle
should change.

---

## Lesson 5 — Speed planning and the acceleration limit

### Why speed and steering cannot be separated

Lateral acceleration is approximately:

```text
a_lateral = speed² × |curvature|
```

Doubling speed quadruples the lateral acceleration needed for the same path.
A steering controller can request the correct geometric curve while the car is
traveling too quickly to follow it safely.

### Curvature-based speed limit

Solving the lateral-acceleration relationship for speed gives:

```text
safe speed = √(lateral acceleration limit / |curvature|)
```

Nearly zero curvature would imply an unlimited speed, so the planner also uses
a straight-line speed cap.

With a `10 m/s²` lateral limit:

| Curvature | Safe speed before cap |
|---:|---:|
| `0.025 m⁻¹` | `20 m/s` |
| `0.10 m⁻¹` | `10 m/s` |
| `0.40 m⁻¹` | `5 m/s` |

### Shared acceleration budget

The simulator checks combined acceleration:

```text
a_net = √(a_longitudinal² + a_lateral²)
```

The hard limit is `12 m/s²`. The controller can use a lower design value, such
as `11 m/s²`, to preserve margin.

When turning already consumes lateral acceleration, the remaining longitudinal
budget is:

```text
a_remaining = √max(a_design² − a_lateral², 0)
```

This expresses a single tire-grip budget shared between turning, acceleration,
and braking.

### Why local speed limits are insufficient

A tight corner may allow only `5 m/s`, while the preceding straight allows
`12 m/s`. The vehicle cannot change speed instantaneously at the corner
boundary. It must begin braking earlier.

Constant-acceleration kinematics gives:

```text
v_current ≤ √(v_next² + 2 × braking × segment_length)
```

A backward pass begins at restrictive future speeds and propagates their effect
to earlier path points. Each earlier entry becomes the largest speed from which
the car can still brake to the next entry under the assumed braking magnitude.

This produces a **braking-feasible speed profile**.

### Feedforward plan and feedback correction

The speed profile is feedforward information: it anticipates known corners.
The speed controller is feedback: it compares planned and measured speed.

```text
speed error = target speed − current speed
requested acceleration = gain × speed error
```

The request is constrained by both:

- The remaining combined-acceleration budget.
- The actuator limits `[-10, 4] m/s²`.

### What “optimal” means here

The braking pass approximates the greatest pointwise speeds satisfying:

- Curvature-based raw speed limits.
- The chosen braking model between neighboring points.

It is not a globally optimal lap-time solution. It does not jointly optimize
the racing line, steering lag, nonlinear vehicle behavior, cone clearance, or
future control inputs. It is an interpretable and useful baseline plan.

#### Implementation shapes

Offline speed planning operates on whole path-aligned arrays:

```text
curvature (N,)
→ lateral-acceleration speed limit at every point
→ raw speed limits (N,)

path (N, 2)
→ displacement to each next wrapped point
→ segment lengths (N,)

raw limits + segment lengths
→ backward braking-feasibility constraint
→ final speed profile (N,)
```

Online speed control uses only the entries relevant to the car's present track
location:

```text
planned speed + measured speed + current curvature
→ proportional acceleration request
→ combined-acceleration budget
→ actuator clipping
→ longitudinal acceleration scalar
```

The offline profile answers “how fast should the car be here?” The online
feedback loop answers “what acceleration should be requested now to approach
that plan?”

---

## Lesson 6 — Complete controller integration

### Precomputation

Track geometry is fixed, so expensive track-wide calculations can happen once:

```text
sample centerline
→ calculate path curvature
→ calculate raw curvature speed limits
→ calculate segment lengths
→ apply backward braking constraints
→ store final speed profile
```

The main aligned arrays are:

```text
path[i]          2D centerline point
curvature[i]     curvature at that point
speed_profile[i] planned speed at that point
```

### One controller call

Every `0.01 s`, the controller performs:

```text
current state
    ↓
nearest centerline index
    ↓
speed-dependent lookahead target
    ↓
target expressed in car coordinates
    ↓
pure-pursuit desired steering
    ↓
steering-rate feedback

current nearest index
    ↓
target speed and current curvature
    ↓
speed feedback with acceleration budget

output = [acceleration, steering_rate]
```

The lookahead target controls steering. Target speed comes from the braking-
adjusted profile at the current nearest index. Future corner information is
already encoded in earlier profile entries by the backward braking pass.

#### Implementation shape

A controller call can be read as two branches sharing the same localization:

```text
state (5,)
→ position, heading, speed, steering angle
→ nearest path index

nearest index + speed + path
→ lookahead target index and target point
→ target in car frame
→ desired steering angle
→ steering-rate command

nearest index + speed profile + curvature + measured speed
→ target speed and acceleration allowance
→ acceleration command

acceleration + steering rate
→ control array (2,)
```

The intermediate indices are bookkeeping that connect the continuous vehicle
position to aligned discrete arrays. They need not be actuator outputs, but the
current index should be retained because it helps the next call localize
continuously.

### Why Lesson 6 returns indices as well as control

The integrated exercise returns:

```text
(control, current_index, target_index)
```

Only `control` is sent into the miniature vehicle model. The two indices make
the controller's internal reasoning visible:

- `current_index` records where the car believes it is on the sampled path and
  becomes `previous_index` on the next simulation call.
- `target_index` records which point ahead the pure-pursuit controller is
  chasing, making lookahead behaviour easier to inspect or plot.

This illustrates a useful distinction between a controller's actuator output
and its bookkeeping or diagnostic output.

### What completing Lesson 6 demonstrates

A successful Lesson 6 run shows that the individual ideas from Lessons 0–5 are
compatible when used repeatedly in a closed loop. The checks ask whether the
car:

```text
makes substantial forward progress
stays close to the circular reference path
keeps net acceleration within the physical limit
keeps both actuator commands within their bounds
```

Passing those checks is evidence that the complete practice pipeline behaves
sensibly on the supplied circle. It is not a claim that the design is optimal
or ready for every possible track; the purpose of the lesson is to give you a
working conceptual foundation that you understand well enough to apply
yourself.
