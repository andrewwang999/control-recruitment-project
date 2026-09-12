"""Small checking and visualization helpers; no exercises live here."""

from __future__ import annotations

import sys
import traceback
import numpy as np


def check_close(name, actual, expected, atol=1e-6):
    """Compare numerical results and report the kind of mismatch."""
    try:
        actual_array = np.asarray(actual, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"{name}\n"
            f"    Your return value could not be interpreted as numbers.\n"
            f"    Received type: {type(actual).__name__}\n"
            f"    Received value: {actual!r}\n"
            f"    Conversion error: {exc}\n"
            f"    Check the function's documented return type and ordering."
        ) from exc

    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name}\n"
            f"    Correct values require the correct shape.\n"
            f"    Expected shape: {expected_array.shape}\n"
            f"    Actual shape:   {actual_array.shape}\n"
            f"    Expected value: {expected_array}\n"
            f"    Actual value:   {actual_array}\n"
            f"    Check whether you need indexing, slicing, or a transpose."
        )

    if not np.allclose(actual_array, expected_array, atol=atol, rtol=0):
        finite = np.isfinite(actual_array) & np.isfinite(expected_array)
        max_error = (
            float(np.max(np.abs(actual_array[finite] - expected_array[finite])))
            if np.any(finite) else float("nan")
        )
        raise AssertionError(
            f"{name}\n"
            f"    The return shape is correct, but one or more values differ.\n"
            f"    Expected: {expected_array}\n"
            f"    Actual:   {actual_array}\n"
            f"    Maximum absolute difference: {max_error:.6g}\n"
            f"    Allowed absolute difference: {atol:.6g}\n"
            f"    Check operand order, signs, axes, and units."
        )
    print(f"  PASS  {name}")


def check_true(name, condition, detail=""):
    if not bool(condition):
        explanation = detail or "The required condition evaluated to False."
        raise AssertionError(
            f"{name}\n"
            f"    {explanation}\n"
            f"    Inspect the named function's return value, type, and shape."
        )
    print(f"  PASS  {name}")


def finish(lesson, message):
    print(f"\nLesson {lesson} complete — {message}")


def wrap_reference(angle):
    """Reference utility used by later lesson checks, not the learner solution."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def circle_path(radius=12.0, count=600):
    """Counter-clockwise circular path, starting at (radius, 0)."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _friendly_exception_hook(exception_type, exception, tb):
    """Add course-specific guidance before Python's normal traceback."""
    frames = traceback.extract_tb(tb)
    exercise_frames = [
        frame for frame in frames
        if "lesson_" in frame.filename and frame.name not in {"run_checks", "<module>"}
    ]
    frame = exercise_frames[-1] if exercise_frames else (frames[-1] if frames else None)

    print("\n" + "=" * 72, file=sys.stderr)
    print("LESSON STOPPED AT THE FIRST FAILED CHECK", file=sys.stderr)
    if frame is not None:
        print(f"Function: {frame.name}()", file=sys.stderr)
        print(f"Location: {frame.filename}:{frame.lineno}", file=sys.stderr)
    print(f"Error:    {exception_type.__name__}: {exception}", file=sys.stderr)

    message = str(exception)
    if exception_type is NotImplementedError:
        hint = "This function is still a TODO. Replace the raise statement with a return value."
    elif exception_type is TypeError and "unsupported operand" in message:
        hint = "An operation received incompatible types. If inputs may be lists, convert them with np.asarray first."
    elif exception_type is TypeError and "NoneType" in message:
        hint = "A function probably reached its end without returning the requested value."
    elif exception_type is ValueError and ("broadcast" in message or "shapes" in message):
        hint = "The array shapes are incompatible. Print each .shape and revisit the documented axes."
    elif exception_type is IndexError:
        hint = "An index is outside the array. Check its shape and remember Python indexing starts at zero."
    elif exception_type is AssertionError:
        hint = "Read the comparison details above; only the first failing scenario is shown."
    else:
        hint = "Read the function's Args/Returns/Direction sections, then inspect intermediate values and shapes."
    print(f"Next step: {hint}", file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)
    sys.__excepthook__(exception_type, exception, tb)


sys.excepthook = _friendly_exception_hook
