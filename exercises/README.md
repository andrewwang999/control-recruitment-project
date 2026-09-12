# Controls Crash Course Exercises

These exercises build the minimum stack needed for the recruitment controller.
They use a small synthetic simulator so you can learn the ideas without
modifying `main.py`.

## Setup

From the repository root:

```bash
python3 -m pip install numpy matplotlib
```

## Workflow

Complete the lessons in order. Replace every `raise NotImplementedError` but do
not edit the checks at the bottom of each file.

[`COURSE_GUIDE.md`](COURSE_GUIDE.md) explains the concepts, mathematics, syntax,
and implementation shapes used throughout Lessons 0–6. Keep it open beside the
current lesson while working.

```bash
python3 exercises/lesson_00_numpy.py
python3 exercises/lesson_01_geometry.py
python3 exercises/lesson_02_dynamics.py
python3 exercises/lesson_03_path.py
python3 exercises/lesson_04_pure_pursuit.py
python3 exercises/lesson_05_speed_planning.py
python3 exercises/lesson_06_integration.py
```

Run all lessons at once with:

```bash
python3 exercises/check_all.py
```

Each lesson prints useful diagnostics when correct. Lesson 06 opens a trajectory
plot of your controller driving around a test track.

## Rules of the game

- Read the docstring, examples, and hints before writing code.
- It is fine to use NumPy documentation.
- Avoid hard-coding the values used by the checks.
- A check passing proves only the tested behavior; think about edge cases too.
- Ask for a conceptual hint before asking for a complete solution.

## Lesson map

0. NumPy essentials for this project
1. Coordinate frames and wrapped angles
2. Bicycle dynamics and numerical integration
3. Sampled paths, progress, tangent, and curvature
4. Pure-pursuit lateral control
5. Acceleration limits and predictive speed planning
6. Complete closed-loop controller on a synthetic track

Lesson 0 is deliberately scoped: it covers the NumPy features used throughout
this course, not every feature in the library.
