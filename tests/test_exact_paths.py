import math
import random
from fractions import Fraction as F
from analysis_app import Window, ImpossibleSequence
from tests.helpers import Case
from tests.oracle import exact


class ExactPaths(Case):
    def test_hand_and_seeded_paths(self):
        rng = random.Random(7821)
        for case in range(24):
            k = 2 + case % 2
            n = case % 7
            def row(size):
                integers = [rng.randrange(4) for _ in range(size)]
                if not any(integers):
                    integers[0] = 1
                return tuple(F(x, sum(integers)) for x in integers)
            initial = row(k)
            transition = tuple(row(k) for _ in range(k))
            emissions = tuple(row(2) for _ in range(k))
            observed = tuple(rng.choice((None, 0, 1)) for _ in range(n))
            names = tuple(str(i) for i in range(k))
            model_id, recording_id = "m" + str(case), "r" + str(case)
            self.app.register_model(model_id, names, ("a", "b"),
                                    tuple(map(float, initial)), tuple(tuple(map(float, r)) for r in transition),
                                    tuple(tuple(map(float, r)) for r in emissions))
            self.app.archive_recording(recording_id, tuple(None if x is None else ("a", "b")[x]
                                                          for x in observed))
            specifications = ((0, n, (0, 1)), (n // 2, n, (1, k - 1 if k > 2 else 0)),
                              (0, 0, (0, 1)))
            windows = tuple(Window(str(i), start, stop, (names[a], names[b]))
                            for i, (start, stop, (a, b)) in enumerate(specifications))
            events = []
            try:
                likelihood, expected, moments = exact(initial, transition, emissions, observed, specifications)
            except ValueError:
                with self.assertRaises(ImpossibleSequence):
                    self.app.analyze(recording_id, model_id, windows, events.append)
                self.assertEqual(events, [])
                continue
            key = self.app.analyze(recording_id, model_id, windows, events.append)
            report = self.app.get_analysis(key)
            self.close_enough(report.log_likelihood, math.log(float(likelihood)))
            self.assertEqual([e.time for e in events], list(range(n)))
            for event, row in zip(events, expected):
                for value, truth in zip(event.probabilities, row):
                    self.close_enough(value, truth)
            for actual, (mean, covariance) in zip(report.reports, moments):
                for a in range(2):
                    self.close_enough(actual.mean[a], mean[a])
                    for b in range(2):
                        self.close_enough(actual.covariance[a][b], covariance[a][b])

    def test_maximum_states(self):
        k = 8
        self.app.register_model("eight", tuple(map(str, range(k))), ("x",), (0.125,) * k,
                                tuple(tuple(float(i == j) for j in range(k)) for i in range(k)),
                                ((1.0,),) * k)
        self.app.archive_recording("eight", (None,) * 3)
        events = []
        key = self.app.analyze("eight", "eight", [Window("w", 0, 3, ("0", "7"))], events.append)
        r = self.app.get_analysis(key).reports[0]
        self.close_enough(r.mean[0], F(3, 8))
        self.close_enough(r.covariance[0][0], F(63, 64))
        self.close_enough(r.covariance[0][1], -F(9, 64))
