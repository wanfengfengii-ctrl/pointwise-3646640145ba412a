"""Bind archives, deliver chronological evidence, then atomically publish a report."""
import json
from dataclasses import dataclass
from .store import Store, encode
from .recordings import Recordings
from .models import Models
from .inference import infer
from .occupancy import Window, Occupancy, validate_windows, summarize
from .counts import (
    CountAnalysis,
    CountCondition,
    constrain,
    counted_reports,
    validate_conditions,
)


@dataclass(frozen=True)
class Evidence:
    time: int
    probabilities: tuple


@dataclass(frozen=True)
class Analysis:
    analysis_id: str
    recording_id: str
    model_id: str
    states: tuple
    length: int
    log_likelihood: float
    reports: tuple


class App:
    _infer = staticmethod(infer)
    _summarize = staticmethod(summarize)

    def __init__(self, path):
        self.store = Store(path)
        self.recordings = Recordings(self.store)
        self.models = Models(self.store)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.store.close()

    def archive_recording(self, recording_id, observations):
        return self.recordings.archive_recording(recording_id, observations)

    def register_model(self, model_id, states, symbols, initial, transitions, emissions):
        return self.models.register_model(model_id, states, symbols, initial, transitions, emissions)

    def analyze(self, recording_id, model_id, windows, evidence_sink):
        if not callable(evidence_sink):
            raise ValueError("evidence_sink must be callable")
        observations = self.recordings.get(recording_id)
        model = self.models.get(model_id)
        windows = validate_windows(windows, len(observations), model.states)
        posterior = None
        try:
            posterior = self._infer(model, observations)
            reports = self._summarize(posterior, windows, model.states)
            for t, row in enumerate(posterior.marginals):
                evidence_sink(Evidence(t, tuple(row)))
            payload = encode({
                "states": model.states, "length": len(observations),
                "log_likelihood": posterior.log_likelihood,
                "reports": [{"window": [r.window.name, r.window.start, r.window.stop,
                                       r.window.states],
                             "mean": r.mean, "covariance": r.covariance} for r in reports]
            })
            with self.store.transaction():
                number = self.store.connection.execute(
                    "SELECT COALESCE(MAX(number),0)+1 FROM analyses"
                ).fetchone()[0]
                self.store.connection.execute(
                    "INSERT INTO analyses VALUES (?,?,?,?)",
                    (number, recording_id, model_id, payload))
            return "analysis-" + str(number)
        finally:
            posterior = None

    def analyze_counts(self, recording_id, model_id, conditions, windows, evidence_sink):
        if not callable(evidence_sink):
            raise ValueError("evidence_sink must be callable")
        observations = self.recordings.get(recording_id)
        model = self.models.get(model_id)
        if len(observations) > 64:
            raise ValueError("analyze_counts supports recordings of at most 64 times")
        conditions = validate_conditions(conditions, len(observations), model.states)
        windows = validate_windows(windows, len(observations), model.states)
        counted = None
        try:
            counted = constrain(model, observations, conditions)
            reports = counted_reports(counted, windows, model.states)
            for t, row in enumerate(counted.marginals):
                evidence_sink(Evidence(t, tuple(row)))
            payload = encode({
                "kind": "counts",
                "states": model.states, "length": len(observations),
                "log_likelihood": counted.log_joint,
                "log_event_probability": counted.log_event_probability,
                "conditions": [[c.start, c.stop, c.state, c.count] for c in conditions],
                "reports": [{"window": [r.window.name, r.window.start, r.window.stop,
                                       r.window.states],
                             "mean": r.mean, "covariance": r.covariance} for r in reports]
            })
            with self.store.transaction():
                number = self.store.connection.execute(
                    "SELECT COALESCE(MAX(number),0)+1 FROM analyses"
                ).fetchone()[0]
                self.store.connection.execute(
                    "INSERT INTO analyses VALUES (?,?,?,?)",
                    (number, recording_id, model_id, payload))
            return "analysis-" + str(number)
        finally:
            counted = None

    def get_analysis(self, analysis_id):
        if not isinstance(analysis_id, str) or not analysis_id.startswith("analysis-"):
            raise KeyError(analysis_id)
        suffix = analysis_id[len("analysis-"):]
        if not suffix.isascii() or not suffix.isdigit() or suffix.startswith("0") or len(suffix) > 18:
            raise KeyError(analysis_id)
        number = int(suffix)
        row = self.store.connection.execute(
            "SELECT recording_id,model_id,payload FROM analyses WHERE number=?", (number,)
        ).fetchone()
        if row is None:
            raise KeyError(analysis_id)
        recording_id, model_id, raw = row
        data = json.loads(raw)
        reports = []
        for item in data["reports"]:
            name, start, stop, states = item["window"]
            reports.append(Occupancy(Window(name, start, stop, tuple(states)),
                                     tuple(item["mean"]), tuple(map(tuple, item["covariance"]))))
        if data.get("kind") == "counts":
            conditions = tuple(
                CountCondition(start, stop, state, count)
                for start, stop, state, count in data["conditions"])
            return CountAnalysis(analysis_id, recording_id, model_id, tuple(data["states"]),
                                 data["length"], data["log_likelihood"], tuple(reports),
                                 conditions, data["log_event_probability"])
        return Analysis(analysis_id, recording_id, model_id, tuple(data["states"]), data["length"],
                        data["log_likelihood"], tuple(reports))
