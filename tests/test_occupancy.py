from analysis_app import Window
from tests.helpers import Case


class Occupation(Case):
    def test_persistent_cross_time_covariance(self):
        self.persistent()
        result, events = self.analyze()
        self.assertEqual([e.time for e in events], [0, 1])
        self.assertEqual(events[0].probabilities, (0.5, 0.5))
        report = result.reports[0]
        self.assertEqual(report.mean, (1.0, 1.0))
        self.assertEqual(report.covariance, ((1.0, -1.0), (-1.0, 1.0)))

    def test_deterministic_alternation_and_overlaps(self):
        self.app.register_model("m", ("off", "on"), ("x",), (1.0, 0.0),
                                ((0.0, 1.0), (1.0, 0.0)), ((1.0,), (1.0,)))
        self.app.archive_recording("r", (None,) * 5)
        result, events = self.analyze((Window("first", 0, 4, ("off", "on")),
                                      Window("last", 1, 5, ("on", "off")),
                                      Window("one", 2, 3, ("off", "on"))))
        self.assertEqual(result.reports[0].mean, (2.0, 2.0))
        self.assertEqual(result.reports[1].mean, (2.0, 2.0))
        self.assertEqual(result.reports[2].mean, (1.0, 0.0))
        for r in result.reports:
            self.assertEqual(r.covariance, ((0.0, 0.0), (0.0, 0.0)))
