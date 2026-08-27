"""Run every exercise's tests against its reference solution, and its skeleton.

The course's own claim about each exercise is that the solution passes the
tests and the untouched skeleton does not, and until this script nothing
checked either one outside a browser. It runs the same harness the app runs
(src/python/harness.py), with the same `course` module the worker registers,
so a test that passes here passes in Pyodide for the same reasons.

Two runs per exercise:

  solution  every test must pass. A failure is a broken exercise.
  skeleton  every test must fail, and the reason must be the skeleton's own
            NotImplementedError rather than a crash in the test itself, which
            is what a learner sees on their first Run tests.

Needs NumPy (Pyodide's only package beyond the standard library):

    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python tools/check_exercises.py
    .venv/bin/python tools/check_exercises.py --verbose   # print every title

Exits non-zero if anything is wrong, so it can gate a change to a test, a
skeleton or a solution.
"""

import argparse
import json
import pathlib
import re
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / "src" / "python"
EX = ROOT / "src" / "exercises"


def load_module(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text(), path.name, "exec"), mod.__dict__)
    return mod


def harness():
    """The app's harness, with `course` registered as the worker does."""
    sys.modules["course"] = load_module(PY / "course_helpers.py", "course")
    return load_module(PY / "harness.py", "harness")


def exercises():
    for d in sorted(EX.iterdir()):
        if (d / "tests.py").exists():
            yield d


def check_registry():
    """The registry's ids against the exercises and the modules.

    src/exercises/registry.ts lists every exercise by id, title, module and
    what it builds, and deliberately imports nothing, because importing an
    exercise pulls its skeleton, tests and solution into the first chunk the
    reader downloads. That is the right trade, and it means the file cannot
    check itself: an id that drifts from an exercise folder silently loses a
    learner's saved work, and a module id that drifts from the module registry
    silently drops the exercise off the front page's outline. Both are checked
    here, from outside the bundle, where importing costs nothing.
    """
    problems = []
    registry = (ROOT / "src" / "exercises" / "registry.ts").read_text()
    modules = (ROOT / "src" / "modules" / "NN" / "index.ts").read_text()

    listed = re.findall(r'\bid: "([a-z0-9-]+)"', registry)
    module_of = dict(zip(listed, re.findall(r'\bmodule: "([a-z0-9]+)"', registry)))
    known_modules = set(re.findall(r'\bid: "([a-z0-9]+)"', modules))

    for ex_id in listed:
        index = EX / ex_id / "index.ts"
        if not index.exists():
            problems.append(f"registry lists {ex_id}, which has no folder in src/exercises")
        elif f'id: "{ex_id}"' not in index.read_text():
            problems.append(f"registry's id {ex_id} does not match the id in its own index.ts")
        if module_of.get(ex_id) not in known_modules:
            problems.append(
                f"registry files {ex_id} under module {module_of.get(ex_id)!r}, "
                "which src/modules/NN/index.ts does not define")

    for d in exercises():
        if d.name not in listed:
            problems.append(
                f"src/exercises/{d.name} exists but the registry does not list it, "
                "so it is invisible on the front page")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="print every test title")
    args = ap.parse_args()

    h = harness()
    problems = []
    for d in exercises():
        tests = (d / "tests.py").read_text()
        solution = (d / "solution.py").read_text()
        skeleton = (d / "skeleton.py").read_text()

        got = json.loads(h.run_exercise(solution, tests))
        if got["setup_error"]:
            problems.append(f"{d.name}: the solution does not even import: {got['setup_error']}")
        failed = [t for t in got["tests"] if not t["passed"]]
        status = "ok" if got["passed"] and not failed else "FAILED"
        print(f"{d.name:16} solution: {len(got['tests']) - len(failed)}/{len(got['tests'])} {status}")
        for t in failed:
            problems.append(f"{d.name}: solution fails {t['name']}: {t['message']}")
        if args.verbose:
            for t in got["tests"]:
                print(f"    {'pass' if t['passed'] else 'FAIL'}  {t['title']}")
        if not got["tests"]:
            problems.append(f"{d.name}: the test file defines no test_ functions")

        # The skeleton must fail the same tests, and fail them by raising the
        # NotImplementedError it ships with: a skeleton that crashes the test
        # itself (a NameError, a bad import) teaches nothing on a first run.
        got = json.loads(h.run_exercise(skeleton, tests))
        if got["setup_error"]:
            problems.append(f"{d.name}: the skeleton does not import: {got['setup_error']}")
            continue
        passed_anyway = [t["name"] for t in got["tests"] if t["passed"]]
        if passed_anyway:
            problems.append(
                f"{d.name}: the skeleton passes {', '.join(passed_anyway)}, "
                "so those tests do not test anything the learner writes")
        wrong_error = [t["name"] for t in got["tests"]
                       if not t["passed"] and "NotImplementedError" not in t["message"]]
        if wrong_error:
            problems.append(
                f"{d.name}: on the untouched skeleton, {', '.join(wrong_error)} fails with "
                "something other than the skeleton's NotImplementedError")
        print(f"{' ':16} skeleton: {len(got['tests']) - len(passed_anyway)}/{len(got['tests'])} "
              f"fail as expected")

    problems += check_registry()
    print(f"{' ':16} registry: {len(list(exercises()))} exercises listed, ids and module ids match")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("every reference solution passes its tests; every skeleton fails them the "
          "way a first run should.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
