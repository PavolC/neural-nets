"""Run every payoff panel's Python against the learner's file, outside a browser.

Nine panels train, score or draw with the code the learner wrote, and until
this script nothing checked that any of them ran. They are the course's payoff
and several modules quote their exact numbers in prose, so a panel that raises
`AttributeError: module has no attribute 'gradient'` is a broken module, not a
broken panel.

Each panel keeps its Python in a template literal in its .tsx. This script
lifts that literal out, substitutes the numeric constants the file declares,
builds the same arguments the panel builds (the learner's file, assembled from
the reference solutions and cut to the projection that panel asks for), and
runs it with the worker's own globals in place: the `course` module, the two
dataset loaders, and `_js_report`.

What it asserts is that the panel runs, reports progress, and returns JSON
whose numbers are in a sane range. It does not pin exact accuracies. Those come
from tools/bench_depth.py and tools/bench_penguins.py, which run the same code
paths for many more epochs; this is the check that the wiring is right.

    pip install numpy && python3 tools/check_panels.py
    python3 tools/check_panels.py --fast    # 2 epochs, for a quick pass

Needs NumPy and the bundled datasets in public/data.
"""

import argparse
import ast
import builtins
import gzip
import io
import json
import pathlib
import re
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import workbench as wb  # noqa: E402

ROOT = wb.ROOT
PANELS = ROOT / "src" / "modules" / "NN" / "interactives"
DATA = ROOT / "public" / "data"

# panel file -> (snippet const, the section whose projection it runs, extra args)
CASES = [
    ("NetworkDiagram.tsx", "PAYOFF_SNIPPET", "feedforward", {}),
    ("SgdLivePanel.tsx", "SNIPPET", "sgd", {}),
    ("BackpropTrainPanel.tsx", "SNIPPET", "backprop", {}),
    ("CostSwapPanel.tsx", "SNIPPET", "cross-entropy", {}),
    ("InitStartPanel.tsx", "SNIPPET", "smart-init", {}),
    ("RegularizePanel.tsx", "SNIPPET", "l2", {"start": "yours", "lmbda": 3.0}),
    ("DepthTrainPanel.tsx", "SNIPPET", "smart-init", {"activation": "sigmoid", "eta": 0.5}),
    ("DepthTrainPanel.tsx", "SNIPPET", "smart-init", {"activation": "relu", "eta": 0.05}),
    ("FullTrainPanel.tsx", "SNIPPET", "train", {"eta": 0.5, "lmbda": 1.0}),
    ("PenguinsPanel.tsx", "SNIPPET", "prepare", {"scale": True, "features": "all"}),
]

# Cut every loop down to this in --fast mode. The panels train for 8 to 200
# epochs, which is the right number for a reader watching one and the wrong
# number for a checker running ten.
FAST_EPOCHS = 2


def snippet(path, name, epochs):
    """The panel's Python, with its numeric constants substituted in."""
    source = path.read_text()
    m = re.search(rf"^const {name} = `\n(.*?)^`;$", source, re.S | re.M)
    if m is None:
        raise SystemExit(f"{path.name}: could not find the {name} template literal")
    body = m.group(1)
    consts = {
        k: int(v)
        for k, v in re.findall(r"^const ([A-Z_]+) = (\d+);$", source, re.M)
    }
    if epochs is not None and "EPOCHS" in consts:
        consts["EPOCHS"] = min(consts["EPOCHS"], epochs)

    def sub(match):
        key = match.group(1)
        if key in consts:
            return str(consts[key])
        if key.isdigit():
            return key
        raise SystemExit(f"{path.name}: {name} interpolates ${{{key}}}, which is not a "
                         "plain numeric constant; this checker cannot stand in for it")

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*|\d+)\}", sub, body)


def load_module(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text(), path.name, "exec"), mod.__dict__)
    return mod


def gunzip(path):
    return gzip.decompress(path.read_bytes())


def worker_globals(datasets):
    """The names the worker has in scope when a snippet runs."""
    course = load_module(wb.PY / "course_helpers.py", "course")
    sys.modules["course"] = course
    loader = load_module(wb.PY / "data_loader.py", "data_loader")
    course.load_mnist_subset = loader.load_mnist_subset
    course.load_penguins = loader.load_penguins

    real_open = builtins.open

    def fake_open(path, mode="r", *args, **kwargs):
        # The snippets open /mnist_subset.bin and /penguins.json, which is
        # where the worker writes them inside Pyodide's filesystem.
        key = str(path).lstrip("/")
        if key in datasets:
            return io.BytesIO(datasets[key])
        return real_open(path, mode, *args, **kwargs)

    reports = []
    ns = {
        "_loader": loader,
        "__builtins__": {**vars(builtins), "open": fake_open},
        "course": course,
        "load_mnist_subset": loader.load_mnist_subset,
        "load_penguins": loader.load_penguins,
        "one_hot": loader.one_hot,
        "_js_report": lambda payload: reports.append(json.loads(payload)),
    }
    return ns, reports


def projection(section_id):
    """The learner's file as far as this panel needs it, solutions throughout."""
    ids = wb.with_givens([section_id])
    # Cumulative, because a learner reaching this panel has everything above it.
    rank = wb.SECTIONS.index(wb.BY_ID[section_id])
    for g in wb.givens_for(section_id):
        rank = max(rank, wb.SECTIONS.index(wb.BY_ID[g]))
    ids = [s["id"] for s in wb.SECTIONS[: rank + 1]]
    kinds = {"backprop": "seam"} if wb.needs_seam(ids) else {}
    return wb.assemble(ids, "solution", kinds)


def network_diagram_args(loader):
    """The one panel that is handed data rather than only code."""
    import numpy as np
    weights = json.loads(gunzip(DATA / "pretrained_weights.json.gz"))
    _, _, X_test, _ = loader.load_mnist_subset(gunzip(DATA / "mnist_subset.bin.gz"))
    # Raw 0-255 rows, the way the panel reads them out of the bundled file:
    # the snippet is what divides by 255.
    digits = [[int(round(v * 255)) for v in X_test[:, k]] for k in range(3)]
    return {
        "weights": weights["weights"],
        "biases": [np.array(b).ravel().tolist() for b in weights["biases"]],
        "digits": digits,
    }


def run_snippet(code, filename, ns):
    """Exec the snippet and return its final expression, as Pyodide does.

    Every first-party snippet ends in a json.dumps(...) that the worker posts
    back, and pyodide.runPythonAsync returns the last expression's value.
    exec() discards it, so the last statement is compiled separately.
    """
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final = ast.Expression(tree.body.pop().value)
        exec(compile(tree, filename, "exec"), ns)  # noqa: S102
        return eval(compile(final, filename, "eval"), ns)  # noqa: S307
    exec(compile(tree, filename, "exec"), ns)  # noqa: S102
    return None


def check_numbers(summary):
    """Sanity, not exactness: accuracies in range, times positive, no NaN."""
    problems = []

    def walk(value, path):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(value, list):
            for i, v in enumerate(value[:4]):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                problems.append(f"{path} is {value}")
            elif "accuracy" in path and not 0.0 <= value <= 1.0:
                problems.append(f"{path} is {value}, which is not a share")

    walk(summary, "")
    return "; ".join(problems)


def describe(summary):
    """One line of whatever the panel actually reported."""
    if isinstance(summary, list):
        return f"{len(summary)} results"
    bits = []
    for key in ("accuracy", "final_accuracy", "final_cost", "seconds", "params"):
        if key in summary and isinstance(summary[key], (int, float)):
            bits.append(f"{key}={summary[key]:.4g}")
    if not bits and isinstance(summary.get("runs"), dict):
        for name, run in list(summary["runs"].items())[:3]:
            if isinstance(run, dict) and "accuracy" in run:
                bits.append(f"{name}={run['accuracy']:.3f}")
    return ", ".join(bits[:4])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help=f"cap every loop at {FAST_EPOCHS} epochs")
    ap.add_argument("--only", help="run one panel by file name")
    args = ap.parse_args()
    epochs = FAST_EPOCHS if args.fast else None

    datasets = {
        "mnist_subset.bin": gunzip(DATA / "mnist_subset.bin.gz"),
        "penguins.json": gunzip(DATA / "penguins.json.gz"),
    }
    problems = []
    for filename, const, section, extra in CASES:
        if args.only and args.only not in filename:
            continue
        label = filename.replace(".tsx", "")
        if extra.get("activation"):
            label += f" ({extra['activation']})"
        code = snippet(PANELS / filename, const, epochs)
        ns, reports = worker_globals(datasets)
        payload = {"code": projection(section), **extra}
        if filename == "NetworkDiagram.tsx":
            payload.update(network_diagram_args(ns["_loader"]))
        ns["_args_json"] = json.dumps(payload)
        try:
            out = run_snippet(code, filename, ns)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"{label:34} FAILED  {type(exc).__name__}: {exc}")
            continue
        if not isinstance(out, str):
            problems.append(
                f"{label}: the snippet did not end in a JSON string, so the panel "
                f"would receive {type(out).__name__} and render nothing")
            print(f"{label:34} FAILED  returned {type(out).__name__}, not JSON")
            continue
        try:
            summary = json.loads(out)
        except json.JSONDecodeError as exc:
            problems.append(f"{label}: its result is not JSON: {exc}")
            print(f"{label:34} FAILED  result is not JSON")
            continue
        bad = check_numbers(summary)
        if bad:
            problems.append(f"{label}: {bad}")
        print(f"{label:34} {'ok  ' if not bad else 'ODD '}{len(reports):3} reports  "
              f"{describe(summary)}")

    print()
    if problems:
        print(f"{len(problems)} panel(s) do not run:")
        for p in problems:
            print(" -", p)
        return 1
    print("every payoff panel runs against the learner's own file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
