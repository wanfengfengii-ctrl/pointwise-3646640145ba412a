import math
from analysis_app import Window, ImpossibleSequence
from tests.helpers import Case


class Conditioning(Case):
    def test_future_hard_evidence_controls_earlier_windows(self):
        self.app.register_model("m", ("off", "on"), ("a", "b"), (0.5, 0.5),
                                ((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)))
        self.app.archive_recording("r", (None, None, None, "b"))
        result, events = self.analyze((Window("early", 0, 2, ("off", "on")),
                                      Window("overlap", 1, 3, ("off", "on"))))
        self.assertEqual([e.probabilities for e in events], [(0.0, 1.0)] * 4)
        for r in result.reports:
            self.assertEqual(r.mean, (0.0, 2.0))
            self.assertEqual(r.covariance, ((0.0, 0.0), (0.0, 0.0)))

    def test_impossible_tail_has_no_evidence_or_report(self):
        self.app.register_model("m", ("off", "on"), ("a", "b"), (1.0, 0.0),
                                ((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)))
        self.app.archive_recording("r", ("a", None, "b"))
        events = []
        with self.assertRaises(ImpossibleSequence):
            self.app.analyze("r", "m", [Window("empty", 0, 0, ("off", "on"))], events.append)
        self.assertEqual(events, [])
        self.no_publication()

    def test_tiny_likelihood_remains_conditioned(self):
        self.app.register_model("m", ("off", "on"), ("tiny", "normal"), (0.5, 0.5),
                                ((1.0, 0.0), (0.0, 1.0)),
                                ((1e-250, 1.0), (2e-250, 1.0)))
        self.app.archive_recording("r", ("tiny",) * 6)
        result, events = self.analyze()
        p = 64.0 / 65.0
        self.assertTrue(math.isfinite(result.log_likelihood))
        self.assertLess(result.log_likelihood, -3000)
        for e in events:
            self.close_enough(e.probabilities[1], p)
        self.close_enough(result.reports[0].mean[1], 6 * p)
        self.close_enough(result.reports[0].covariance[1][1], 36 * p * (1 - p))

    def test_full_missing_and_empty_recording(self):
        self.persistent(())
        result, events = self.analyze()
        self.assertEqual(events, [])
        self.assertEqual(result.log_likelihood, 0.0)
        self.assertEqual(result.reports[0].mean, (0.0, 0.0))
