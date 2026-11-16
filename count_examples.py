"""Public input fixtures and hand-computed expectations; no count inference implementation."""
import argparse
import copy
import json
import math
from analysis_app import App, Window


CASES = {
    "once": {
        "n": 3,
        "conditions": ((0, 3, "on", 1),),
        "windows": (("all", 0, 3, ("on", "off")), ("early", 0, 2, ("on", "off"))),
    },
    "overlap": {
        "n": 4,
        "conditions": ((0, 3, "on", 1), (1, 4, "on", 1)),
        "windows": (("all", 0, 4, ("on", "off")),),
    },
    "conflict": {
        "n": 3,
        "conditions": ((0, 3, "on", 0), (0, 3, "on", 1)),
        "windows": (("all", 0, 3, ("on", "off")),),
    },
    "empty": {
        "n": 0,
        "conditions": ((0, 0, "on", 0),),
        "windows": (("all", 0, 0, ("on", "off")),),
    },
    "rare": {
        "n": 2,
        "conditions": ((0, 2, "off", 2),),
        "windows": (("all", 0, 2, ("on", "off")),),
    },
}

EXPECTED = {
    "once": {
        "event_probability": [3, 8],
        "marginals": [[[2, 3], [1, 3]]] * 3,
        "reports": [
            {"mean": [[1, 1], [2, 1]], "covariance": [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]},
            {"mean": [[2, 3], [4, 3]], "covariance": [[[2, 9], [-2, 9]], [[-2, 9], [2, 9]]]},
        ],
    },
    "overlap": {
        "event_probability": [3, 16],
        "marginals": [[[2, 3], [1, 3]]] * 4,
        "reports": [
            {"mean": [[4, 3], [8, 3]], "covariance": [[[2, 9], [-2, 9]], [[-2, 9], [2, 9]]]},
        ],
    },
    "conflict": {"event_probability": [0, 1], "error": "ImpossibleSequence"},
    "empty": {
        "event_probability": [1, 1], "marginals": [],
        "reports": [{"mean": [[0, 1], [0, 1]], "covariance": [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]}],
    },
    "rare": {
        "log_event_probability": 2 * math.log(1e-250),
        "marginals": [[[1, 1], [0, 1]]] * 2,
        "reports": [{"mean": [[0, 1], [2, 1]], "covariance": [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]}],
    },
}


def load_case(app, name):
    case = CASES[name]
    initial = (1e-250, 1.0) if name == "rare" else (0.5, 0.5)
    app.register_model(name + "-model", ("off", "on"), ("x",), initial,
                       (initial, initial), ((1.0,), (1.0,)))
    app.archive_recording(name + "-recording", (None,) * case["n"])
    return tuple(case["conditions"]), tuple(Window(*w) for w in case["windows"])


def expected_case(name):
    return copy.deepcopy(EXPECTED[name])


def check_controls():
    summaries = []
    with App(":memory:") as app:
        for name, case in CASES.items():
            specs, windows = load_case(app, name)
            assert 1 <= len(specs) <= 2
            for start, stop, state, count in specs:
                assert 0 <= start <= stop <= case["n"]
                assert state in ("off", "on") and 0 <= count <= min(8, stop - start)
            seen = []
            key = app.analyze(name + "-recording", name + "-model", windows, seen.append)
            result = app.get_analysis(key)
            assert len(seen) == case["n"]
            assert [event.time for event in seen] == list(range(case["n"]))
            assert math.isfinite(result.log_likelihood)
            for report in result.reports:
                n = report.window.stop - report.window.start
                if name == "rare":
                    continue
                assert report.mean == (n / 2, n / 2)
                assert report.covariance == ((n / 4, -n / 4), (-n / 4, n / 4))
            summaries.append({"case": name, "ordinary_mean": result.reports[0].mean,
                              "count_expectation": expected_case(name)})
    print(json.dumps({"ordinary_controls": summaries}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check ordinary baseline controls only")
    args = parser.parse_args()
    if args.check:
        check_controls()
    else:
        print(json.dumps({"cases": CASES, "expected": EXPECTED}, sort_keys=True))


if __name__ == "__main__":
    main()
