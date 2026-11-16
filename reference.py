"""Fixed normal materialized backend, retained for result and resource comparison."""
from analysis_app.publication import App
from analysis_app.inference import materialize
from analysis_app.occupancy import materialized_reports


class ReferenceApp(App):
    _infer = staticmethod(materialize)
    _summarize = staticmethod(materialized_reports)
