"""Run an actual archive -> analyze -> evidence -> published report -> reopen cycle."""
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from analysis_app import App
from samples import load_demo


def main():
    with tempfile.TemporaryDirectory(prefix="state-analysis-demo-") as directory:
        path = Path(directory) / "archive.sqlite"
        with App(path) as app:
            windows = load_demo(app)
            def show(event):
                print(json.dumps({"time": event.time, "posterior": event.probabilities}))
            analysis_id = app.analyze("shift", "equipment", windows, show)
        with App(path) as app:
            print(json.dumps(asdict(app.get_analysis(analysis_id)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
