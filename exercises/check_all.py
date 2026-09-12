"""Run every completed lesson in a fresh Python process."""

import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
LESSONS = [
    "lesson_00_numpy.py",
    "lesson_01_geometry.py",
    "lesson_02_dynamics.py",
    "lesson_03_path.py",
    "lesson_04_pure_pursuit.py",
    "lesson_05_speed_planning.py",
    "lesson_06_integration.py",
]


def main():
    passed = 0
    for lesson in LESSONS:
        path = os.path.join(HERE, lesson)
        print(f"\n{'=' * 68}\nRunning {lesson}\n{'=' * 68}")
        result = subprocess.run([sys.executable, path, "--no-plot"], cwd=os.path.dirname(HERE))
        if result.returncode != 0:
            print(f"\nStopped at {lesson}. Complete its TODOs, then run this again.")
            return result.returncode
        passed += 1
    print(f"\nAll {passed} lessons passed. You are ready to start cooking in main.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
