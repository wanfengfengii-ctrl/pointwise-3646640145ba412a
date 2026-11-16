import math
import unittest
from analysis_app import App, Window


class Case(unittest.TestCase):
    def setUp(self):
        self.app = App(":memory:")
        self.addCleanup(self.app.close)

    def persistent(self, observations=(None, None), emissions=((1.0,), (1.0,))):
        self.app.register_model("m", ("off", "on"), ("x",), (0.5, 0.5),
                                ((1.0, 0.0), (0.0, 1.0)), emissions)
        self.app.archive_recording("r", observations)

    def analyze(self, windows=None):
        events = []
        if windows is None:
            n = len(self.app.recordings.get("r"))
            windows = (Window("all", 0, n, ("off", "on")),)
        key = self.app.analyze("r", "m", windows, events.append)
        return self.app.get_analysis(key), events

    def close_enough(self, actual, expected, abs_tol=2e-10, rel_tol=2e-10):
        self.assertTrue(math.isclose(actual, float(expected), abs_tol=abs_tol, rel_tol=rel_tol),
                        (actual, expected))

    def no_publication(self):
        self.assertEqual(self.app.store.connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0], 0)
        self.assertFalse(self.app.store.connection.in_transaction)
