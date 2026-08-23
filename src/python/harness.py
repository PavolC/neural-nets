"""Exercise test harness. Runs inside Pyodide.

run_exercise(learner_code, tests_code) executes the learner's code as a
fresh `submission` module, then runs every `test_*` function defined in
tests_code, in definition order. Returns a JSON string:

    {
      "setup_error": null | {"message": str, "line": int | null},
      "tests": [{"name": str, "title": str, "passed": bool, "message": str}],
      "passed": bool
    }

Test functions signal failure by raising AssertionError with a teaching
message (see CLAUDE.md: failure messages are teaching content). Any other
exception is reported with its type, message, and the line in the learner's
code that raised it, when that can be determined.
"""

import json
import sys
import traceback
import types

LEARNER_FILENAME = "your_code.py"


def _learner_line(exc):
    """Deepest traceback line inside the learner's code, or None."""
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == LEARNER_FILENAME:
            line = frame.lineno
    return line


def _format_error(exc):
    message = f"{type(exc).__name__}: {exc}"
    line = _learner_line(exc)
    if isinstance(exc, SyntaxError) and exc.filename == LEARNER_FILENAME:
        line = exc.lineno
    return {"message": message, "line": line}


def run_scratch(learner_code):
    """Execute the learner's code alone, no tests: for printing and playing.

    Anything printed streams to the UI's output panel via the worker's
    stdout handler. Returns JSON: {"error": null | {"message", "line"}}.
    """
    scratch = types.ModuleType("scratch")
    scratch.__file__ = LEARNER_FILENAME
    try:
        exec(compile(learner_code, LEARNER_FILENAME, "exec"), scratch.__dict__)
    except Exception as exc:
        return json.dumps({"error": _format_error(exc)})
    return json.dumps({"error": None})


def run_exercise(learner_code, tests_code):
    results = {"setup_error": None, "tests": [], "passed": False}

    submission = types.ModuleType("submission")
    submission.__file__ = LEARNER_FILENAME
    try:
        exec(compile(learner_code, LEARNER_FILENAME, "exec"), submission.__dict__)
    except Exception as exc:
        results["setup_error"] = _format_error(exc)
        return json.dumps(results)
    sys.modules["submission"] = submission

    test_ns = {}
    exec(compile(tests_code, "tests.py", "exec"), test_ns)
    test_fns = [(name, fn) for name, fn in test_ns.items()
                if name.startswith("test_") and callable(fn)]

    all_passed = True
    for name, fn in test_fns:
        title = (fn.__doc__ or name.replace("_", " ")).strip().splitlines()[0]
        entry = {"name": name, "title": title, "passed": True, "message": ""}
        try:
            fn()
        except AssertionError as exc:
            entry["passed"] = False
            entry["message"] = str(exc) or "assertion failed (no message)"
        except Exception as exc:
            entry["passed"] = False
            err = _format_error(exc)
            where = f" (raised at {LEARNER_FILENAME} line {err['line']})" if err["line"] else ""
            entry["message"] = (
                f"your code raised {err['message']}{where}. "
                "The test called your function with valid inputs, so the "
                "error is in the implementation, not the test."
            )
        all_passed = all_passed and entry["passed"]
        results["tests"].append(entry)

    results["passed"] = all_passed and bool(test_fns)
    return json.dumps(results)
