"""Frozen end-to-end normal baseline measurements, with sequential evidence fixtures."""
import argparse
import ast
import gc
import hashlib
import json
import math
import statistics
import struct
import tempfile
import time
import tracemalloc
from pathlib import Path
from analysis_app import App
from reference import ReferenceApp
from samples import load_workload

ROOT = Path(__file__).resolve().parent
ABS_TOL = 2e-9
REL_TOL = 2e-9
FROZEN = {
    "analysis_app/inference.py": ("log", "logsum", "probabilities", "materialize"),
    "analysis_app/occupancy.py": ("materialized_reports",),
    "reference.py": ("ReferenceApp",),
}


def frozen_digests():
    result = {}
    for path, names in FROZEN.items():
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                normalized = ast.dump(node, include_attributes=False)
                result[path + ":" + node.name] = hashlib.sha256(normalized.encode()).hexdigest()
    return result


def verify_frozen():
    expected = json.loads((ROOT / "baseline-lock.json").read_text(encoding="utf-8"))
    if frozen_digests() != expected:
        raise AssertionError("materialized reference or numerical reference helpers changed")


def near(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=ABS_TOL, rel_tol=REL_TOL):
        raise AssertionError((actual, expected))


class EvidenceFixture:
    def __init__(self, handle, n, k, writing):
        self.handle, self.n, self.k, self.writing = handle, n, k, writing
        self.position = 0
        self.record = struct.Struct("<" + "d" * k)

    def __call__(self, event):
        if event.time != self.position or len(event.probabilities) != self.k:
            raise AssertionError("evidence order or dimensions")
        if self.writing:
            self.handle.write(self.record.pack(*event.probabilities))
        else:
            raw = self.handle.read(self.record.size)
            if len(raw) != self.record.size:
                raise AssertionError("missing reference evidence")
            for actual, expected in zip(event.probabilities, self.record.unpack(raw)):
                near(actual, expected)
        self.position += 1

    def finish(self):
        if self.position != self.n:
            raise AssertionError("missing evidence")
        if not self.writing and self.handle.read(1):
            raise AssertionError("unconsumed reference evidence")


def compare_reports(actual, expected):
    if (actual.recording_id, actual.model_id, actual.states, actual.length) != (
            expected.recording_id, expected.model_id, expected.states, expected.length):
        raise AssertionError("binding mismatch")
    near(actual.log_likelihood, expected.log_likelihood)
    if len(actual.reports) != len(expected.reports):
        raise AssertionError("report count")
    for a, e in zip(actual.reports, expected.reports):
        if a.window != e.window:
            raise AssertionError("window mismatch")
        for i in range(2):
            near(a.mean[i], e.mean[i])
            for j in range(2):
                near(a.covariance[i][j], e.covariance[i][j])


def measure(app, windows, sink):
    # Fixture open, database creation and ingestion happen before this boundary.
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        key = app.analyze("recording", "model", windows, sink)
        result = app.get_analysis(key)
        sink.finish()
        seconds = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {"seconds": seconds, "python_peak_bytes": peak}


def run_suite(mode, sizes=(10000, 40000), rounds=3):
    verify_frozen()
    records = []
    with tempfile.TemporaryDirectory(prefix="state-analysis-bench-") as temporary:
        directory = Path(temporary)
        for family in ("dense", "sparse"):
            for n in sizes:
                for repeat in range(rounds):
                    serial = family + "-" + str(n) + "-" + str(repeat)
                    fixture = directory / (serial + ".evidence")
                    with ReferenceApp(directory / (serial + "-reference.sqlite")) as app:
                        windows = load_workload(app, family, n)
                        k = len(app.models.get("model").states)
                        with fixture.open("wb") as handle:
                            expected, reference_metrics = measure(
                                app, windows, EvidenceFixture(handle, n, k, True))
                    record = {"family": family, "n": n, "k": k, "round": repeat,
                              "reference": reference_metrics}
                    # The frozen normal backend creates no self-owned temporary files.
                    record["reference"]["temporary_peak_bytes"] = 0
                    if mode == "compare":
                        with App(directory / (serial + "-production.sqlite")) as app:
                            windows = load_workload(app, family, n)
                            with fixture.open("rb") as handle:
                                actual, metrics = measure(
                                    app, windows, EvidenceFixture(handle, n, k, False))
                            compare_reports(actual, expected)
                        # Arbitrary future implementations require independent platform disk metering.
                        metrics["temporary_peak_bytes"] = None
                        record["production"] = metrics
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    fixture.unlink()
    summary = []
    for family in ("dense", "sparse"):
        for n in sizes:
            group = [r for r in records if r["family"] == family and r["n"] == n]
            entry = {"family": family, "n": n}
            for name in ("reference", "production") if mode == "compare" else ("reference",):
                entry[name] = {
                    "median_seconds": statistics.median(r[name]["seconds"] for r in group),
                    "median_python_peak_bytes": statistics.median(r[name]["python_peak_bytes"] for r in group)}
            summary.append(entry)
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "compare"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_suite(args.mode, sizes=(32, 128) if args.smoke else (10000, 40000),
              rounds=1 if args.smoke else 3)


if __name__ == "__main__":
    main()
