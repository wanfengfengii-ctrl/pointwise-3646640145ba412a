"""Count-event demo: condition evidence and window uncertainty on one event.

A three-time recording of independent fair off/on states is analyzed twice on
the same archive and model: the ordinary posterior and the posterior
conditioned on "on occurs exactly once across the whole recording".  The event
changes both the chronological evidence and the full window covariance; the
published CountAnalysis is then read back after closing the database.
"""
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from analysis_app import App, CountCondition, Window


def main():
    with tempfile.TemporaryDirectory(prefix="count-analysis-demo-") as directory:
        path = Path(directory) / "archive.sqlite"
        with App(path) as app:
            app.register_model(
                "sensor", ("off", "on"), ("x",), (0.5, 0.5),
                ((0.5, 0.5), (0.5, 0.5)), ((1.0,), (1.0,)))
            app.archive_recording("trace", (None, None, None))
            window = Window("whole", 0, 3, ("on", "off"))

            ordinary_events = []
            ordinary_key = app.analyze("trace", "sensor", (window,), ordinary_events.append)
            ordinary = app.get_analysis(ordinary_key)

            condition = CountCondition(0, 3, "on", 1)
            counted_events = []
            counted_key = app.analyze_counts(
                "trace", "sensor", (condition,), (window,), counted_events.append)
            counted = app.get_analysis(counted_key)

            print("ordinary evidence:",
                  [event.probabilities for event in ordinary_events])
            print("conditioned evidence (on exactly once):",
                  [event.probabilities for event in counted_events])
            print("ordinary report:", json.dumps(asdict(ordinary), ensure_ascii=False))
            print("conditioned report:", json.dumps(asdict(counted), ensure_ascii=False))
            print("log P(O,E) =", counted.log_likelihood,
                  " log P(E|O) =", counted.log_event_probability)

        with App(path) as app:
            reopened = app.get_analysis(counted_key)
            print("reopened:", type(reopened).__name__)
            print(json.dumps(asdict(reopened), ensure_ascii=False, indent=2))
            assert reopened == counted
            assert app.get_analysis(ordinary_key) == ordinary


if __name__ == "__main__":
    main()
