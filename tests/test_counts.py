import math
import random
import sqlite3
import tempfile
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path

from analysis_app import (
    App, Window, CountCondition, CountAnalysis, Analysis, ImpossibleSequence,
)
from count_examples import CASES, EXPECTED, load_case
from tests.helpers import Case
from tests.count_oracle import exact_counts

ABS_TOL = REL_TOL = 2e-10


def close(test, actual, expected):
    test.assertTrue(math.isclose(actual, float(expected), abs_tol=ABS_TOL, rel_tol=REL_TOL),
                    (actual, expected))


def count_rows(app):
    return app.store.connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]


def fraction_log(value):
    return -math.inf if value == 0 else math.log(value.numerator) - math.log(value.denominator)


class CountHandExamples(Case):
    def test_published_hand_examples(self):
        for name in CASES:
            specs, windows = load_case(self.app, name)
            conditions = tuple(CountCondition(*spec) for spec in specs)
            events = []
            if name == "conflict":
                before = count_rows(self.app)
                with self.assertRaises(ImpossibleSequence):
                    self.app.analyze_counts(name + "-recording", name + "-model",
                                            conditions, windows, events.append)
                self.assertEqual(events, [])
                self.assertEqual(count_rows(self.app), before)
                continue
            key = self.app.analyze_counts(name + "-recording", name + "-model",
                                          conditions, windows, events.append)
            result = self.app.get_analysis(key)
            self.assertIsInstance(result, CountAnalysis)
            self.assertEqual(result.conditions, conditions)
            expected = EXPECTED[name]
            if "event_probability" in expected:
                event = F(*expected["event_probability"])
                close(self, result.log_event_probability, math.log(event))
                close(self, result.log_likelihood, math.log(event))
            else:
                self.assertTrue(math.isclose(result.log_event_probability,
                                             expected["log_event_probability"], rel_tol=REL_TOL))
                self.assertTrue(math.isfinite(result.log_likelihood))
            self.assertEqual([e.time for e in events], list(range(CASES[name]["n"])))
            for event, wanted in zip(events, expected["marginals"]):
                self.assertEqual(len(event.probabilities), len(result.states))
                for value, rational in zip(event.probabilities, wanted):
                    close(self, value, F(*rational))
            self.assertEqual(len(result.reports), len(expected["reports"]))
            for report, wanted in zip(result.reports, expected["reports"]):
                for a in range(2):
                    close(self, report.mean[a], F(*wanted["mean"][a]))
                    for b in range(2):
                        close(self, report.covariance[a][b], F(*wanted["covariance"][a][b]))

    def test_rare_positive_mass_is_not_rejected(self):
        specs, windows = load_case(self.app, "rare")
        conditions = tuple(CountCondition(*spec) for spec in specs)
        key = self.app.analyze_counts("rare-recording", "rare-model", conditions,
                                      windows, lambda event: None)
        result = self.app.get_analysis(key)
        self.assertTrue(math.isfinite(result.log_likelihood))
        self.assertTrue(math.isfinite(result.log_event_probability))
        # The joint probability itself is far below the floating-point minimum.
        self.assertEqual(math.exp(result.log_likelihood), 0.0)
        self.assertLess(result.log_likelihood, -1000.0)
        self.assertEqual(result.reports[0].mean, (0.0, 2.0))
        self.assertEqual(result.reports[0].covariance, ((0.0, 0.0), (0.0, 0.0)))


class CountExactPaths(Case):
    def _random_model(self, tag, k, rng):
        def row(size):
            integers = [rng.randrange(4) for _ in range(size)]
            if not any(integers):
                integers[0] = 1
            return tuple(F(x, sum(integers)) for x in integers)
        initial = row(k)
        transition = tuple(row(k) for _ in range(k))
        emissions = tuple(row(2) for _ in range(k))
        names = tuple("s" + str(i) for i in range(k))
        self.app.register_model(
            tag, names, ("a", "b"), tuple(map(float, initial)),
            tuple(tuple(map(float, r)) for r in transition),
            tuple(tuple(map(float, r)) for r in emissions))
        return names, initial, transition, emissions

    def test_seeded_random_paths_against_enumeration(self):
        rng = random.Random(91357)
        for case in range(20):
            k = 2 + case % 3
            n = case % 7
            tag = "case" + str(case)
            names, initial, transition, emissions = self._random_model(tag, k, rng)
            observed = tuple(rng.choice((None, 0, 1)) for _ in range(n))
            self.app.archive_recording(
                tag, tuple(None if x is None else ("a", "b")[x] for x in observed))
            condition_specs = []
            state_a = rng.randrange(k)
            condition_specs.append((0, n, state_a, rng.randrange(min(8, n) + 1) if n else 0))
            if case % 2:
                lo, hi = sorted(rng.sample(range(n + 1), 2)) if n else (0, 0)
                condition_specs.append((lo, hi, rng.randrange(k),
                                        rng.randrange(min(8, hi - lo) + 1) if hi > lo else 0))
            conditions = tuple(CountCondition(start, stop, names[state], count)
                               for start, stop, state, count in condition_specs)
            window_specs = [(0, n, (0, k - 1)),
                            (n // 2, n, (state_a, (state_a + 1) % k))]
            if n >= 2:
                window_specs.append((0, 0, (0, 1)))
            windows = tuple(Window(str(i), start, stop, (names[a], names[b]))
                            for i, (start, stop, (a, b)) in enumerate(window_specs))
            events = []
            try:
                joint, plain, expected_marginals, expected_moments = exact_counts(
                    initial, transition, emissions, observed, condition_specs,
                    [(start, stop, targets) for start, stop, targets in window_specs])
            except ValueError:
                before = count_rows(self.app)
                with self.assertRaises(ImpossibleSequence):
                    self.app.analyze_counts(tag, tag, conditions, windows, events.append)
                self.assertEqual(events, [])
                self.assertEqual(count_rows(self.app), before)
                continue
            key = self.app.analyze_counts(tag, tag, conditions, windows, events.append)
            result = self.app.get_analysis(key)
            close(self, result.log_likelihood, fraction_log(joint))
            close(self, result.log_event_probability, fraction_log(joint / plain))
            for event, wanted in zip(events, expected_marginals):
                for value, rational in zip(event.probabilities, wanted):
                    close(self, value, rational)
            for report, (mean, covariance) in zip(result.reports, expected_moments):
                for a in range(2):
                    close(self, report.mean[a], mean[a])
                    for b in range(2):
                        close(self, report.covariance[a][b], covariance[a][b])

    def test_structural_zeros_and_zero_count_condition(self):
        # State 0 can never emit "b"; deterministic self-transitions create
        # structural zeros in the transition matrix.
        self.app.register_model(
            "m", ("off", "on"), ("a", "b"), (1.0, 0.0),
            ((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)))
        self.app.archive_recording("r", ("a", None, None, "a"))
        conditions = (CountCondition(0, 4, "on", 0),)
        events = []
        key = self.app.analyze_counts("r", "m", conditions,
                                      (Window("w", 0, 4, ("off", "on")),), events.append)
        result = self.app.get_analysis(key)
        self.assertEqual([e.probabilities for e in events], [(1.0, 0.0)] * 4)
        self.assertEqual(result.reports[0].mean, (4.0, 0.0))
        self.assertEqual(result.reports[0].covariance, ((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual(result.log_event_probability, 0.0)

    def test_tail_hard_evidence_changes_conditioned_windows(self):
        self.app.register_model(
            "m", ("off", "on"), ("a", "b"), (0.5, 0.5),
            ((0.8, 0.2), (0.3, 0.7)), ((1.0, 0.0), (0.0, 1.0)))
        self.app.archive_recording("r", (None, None, None, "b"))
        conditions = (CountCondition(0, 4, "on", 2),)
        windows = (Window("early", 0, 2, ("off", "on")),
                   Window("outside", 1, 3, ("on", "off")))
        key = self.app.analyze_counts("r", "m", conditions, windows, lambda event: None)
        result = self.app.get_analysis(key)
        # Enumerate the same model independently to lock every reported number.
        initial = (F(1, 2), F(1, 2))
        transition = ((F(4, 5), F(1, 5)), (F(3, 10), F(7, 10)))
        emissions = ((F(1), F(0)), (F(0), F(1)))
        joint, plain, marginals, moments = exact_counts(
            initial, transition, emissions, (None, None, None, 1),
            ((0, 4, 1, 2),),
            ((0, 2, (0, 1)), (1, 3, (1, 0))))
        close(self, result.log_likelihood, fraction_log(joint))
        close(self, result.log_event_probability, fraction_log(joint / plain))
        for report, (mean, covariance) in zip(result.reports, moments):
            for a in range(2):
                close(self, report.mean[a], mean[a])
                for b in range(2):
                    close(self, report.covariance[a][b], covariance[a][b])

    def test_duplicate_conditions_counted_once(self):
        self.app.register_model(
            "m", ("off", "on"), ("x",), (0.5, 0.5),
            ((0.5, 0.5), (0.5, 0.5)), ((1.0,), (1.0,)))
        self.app.archive_recording("r", (None,) * 3)
        duplicate = (CountCondition(0, 3, "on", 1), CountCondition(0, 3, "on", 1))
        single = (CountCondition(0, 3, "on", 1),)
        windows = (Window("w", 0, 3, ("on", "off")),)
        first = self.app.get_analysis(
            self.app.analyze_counts("r", "m", single, windows, lambda event: None))
        second = self.app.get_analysis(
            self.app.analyze_counts("r", "m", duplicate, windows, lambda event: None))
        self.assertEqual(second.conditions, duplicate)
        self.assertTrue(math.isclose(second.log_event_probability,
                                     first.log_event_probability, abs_tol=ABS_TOL))
        self.assertEqual(second.reports, first.reports)


class CountInputs(Case):
    def setUp(self):
        super().setUp()
        self.app.register_model("m", ("off", "on"), ("x",), (0.5, 0.5),
                                ((0.5, 0.5), (0.5, 0.5)), ((1.0,), (1.0,)))
        self.app.archive_recording("r", (None,) * 4)

    def test_invalid_conditions_deliver_nothing(self):
        bads = [
            (),
            (CountCondition(0, 4, "on", 1),) * 3,
            (("w", 0, 4, 1),),
            (CountCondition(-1, 4, "on", 1),),
            (CountCondition(3, 2, "on", 1),),
            (CountCondition(0, 5, "on", 1),),
            (CountCondition(0, 4, "unknown", 1),),
            (CountCondition(0, 4, "on", -1),),
            (CountCondition(0, 4, "on", 5),),
            (CountCondition(0, 2, "on", 3),),
            (CountCondition(True, 4, "on", 1),),
            (CountCondition(0, 4, "on", True),),
            (CountCondition(0, 0, "on", 1),),
        ]
        for conditions in bads:
            events = []
            with self.assertRaises(ValueError):
                self.app.analyze_counts("r", "m", conditions, (), events.append)
            self.assertEqual(events, [])
            self.no_publication()

    def test_length_limit_and_unknown_bindings(self):
        self.app.archive_recording("long", (None,) * 65)
        with self.assertRaises(ValueError):
            self.app.analyze_counts("long", "m", (CountCondition(0, 65, "on", 1),),
                                    (), lambda event: None)
        for recording, model in (("missing", "m"), ("r", "missing")):
            with self.assertRaises(KeyError):
                self.app.analyze_counts(recording, model,
                                        (CountCondition(0, 1, "on", 0),), (),
                                        lambda event: None)
        with self.assertRaises(ValueError):
            self.app.analyze_counts("r", "m", (CountCondition(0, 1, "on", 0),),
                                    (), None)

    def test_condition_snapshot_and_input_mutation(self):
        conditions = [CountCondition(0, 4, "on", 1)]
        key = self.app.analyze_counts("r", "m", conditions,
                                      (Window("w", 0, 4, ("on", "off")),),
                                      lambda event: None)
        conditions.append(CountCondition(0, 4, "off", 2))
        result = self.app.get_analysis(key)
        self.assertEqual(result.conditions, (CountCondition(0, 4, "on", 1),))
        with self.assertRaises(FrozenInstanceError):
            result.conditions[0].count = 2

    def test_windows_still_use_original_rules(self):
        bad_windows = [Window("w", -1, 2, ("off", "on")),
                       Window("w", 0, 5, ("off", "on")),
                       Window("w", 0, 1, ("off", "other"))]
        for window in bad_windows:
            with self.assertRaises(ValueError):
                self.app.analyze_counts("r", "m", (CountCondition(0, 4, "on", 1),),
                                        [window], lambda event: None)
        good = self.app.get_analysis(self.app.analyze_counts(
            "r", "m", (CountCondition(0, 4, "on", 1),),
            (Window("empty", 2, 2, ("off", "on")),), lambda event: None))
        self.assertEqual(good.reports[0].mean, (0.0, 0.0))


class CountPublication(Case):
    def test_sink_exception_and_retry_keeps_id_space(self):
        self.app.register_model("m", ("off", "on"), ("x",), (0.5, 0.5),
                                ((0.5, 0.5), (0.5, 0.5)), ((1.0,), (1.0,)))
        self.app.archive_recording("r", (None,) * 4)
        marker = RuntimeError("receiver disconnected")
        seen = []
        def receiver(event):
            seen.append(event.time)
            if event.time == 1:
                raise marker
        with self.assertRaises(RuntimeError) as raised:
            self.app.analyze_counts("r", "m", (CountCondition(0, 4, "on", 1),),
                                    (), receiver)
        self.assertIs(raised.exception, marker)
        self.assertEqual(seen, [0, 1])
        self.no_publication()
        key = self.app.analyze_counts("r", "m", (CountCondition(0, 4, "on", 1),),
                                      (), lambda event: None)
        self.assertEqual(key, "analysis-1")
        self.assertIsInstance(self.app.get_analysis(key), CountAnalysis)

    def test_database_abort_preserves_old_reports(self):
        self.app.register_model("m", ("off", "on"), ("x",), (0.5, 0.5),
                                ((0.5, 0.5), (0.5, 0.5)), ((1.0,), (1.0,)))
        self.app.archive_recording("r", (None,) * 3)
        ordinary = self.app.get_analysis(
            self.app.analyze("r", "m", (), lambda event: None))
        self.assertEqual(ordinary.analysis_id, "analysis-1")
        self.app.store.connection.execute("""
            CREATE TRIGGER reject_publication BEFORE INSERT ON analyses
            BEGIN SELECT RAISE(ABORT,'publication storage rejected insert'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.app.analyze_counts("r", "m", (CountCondition(0, 3, "on", 1),),
                                    (), lambda event: None)
        self.assertEqual(self.app.get_analysis("analysis-1"), ordinary)
        self.assertIsInstance(self.app.get_analysis("analysis-1"), Analysis)
        with self.assertRaises(KeyError):
            self.app.get_analysis("analysis-2")
        self.app.store.connection.execute("DROP TRIGGER reject_publication")
        key = self.app.analyze_counts("r", "m", (CountCondition(0, 3, "on", 1),),
                                      (), lambda event: None)
        self.assertEqual(key, "analysis-2")
        self.assertIsInstance(self.app.get_analysis(key), CountAnalysis)

    def test_reopen_real_file_returns_count_analysis(self):
        with tempfile.TemporaryDirectory(prefix="count-analysis-test-") as directory:
            path = Path(directory) / "archive.sqlite"
            with App(path) as app:
                specs, windows = load_case(app, "overlap")
                conditions = tuple(CountCondition(*spec) for spec in specs)
                key = app.analyze_counts("overlap-recording", "overlap-model",
                                         conditions, windows, lambda event: None)
                original = app.get_analysis(key)
            with App(path) as app:
                reopened = app.get_analysis(key)
                self.assertEqual(reopened, original)
                self.assertIsInstance(reopened, CountAnalysis)
                self.assertEqual(reopened.conditions, conditions)
                for report in reopened.reports:
                    self.assertIsInstance(report.window.states, tuple)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["archive.sqlite"])


class CountLargeShape(Case):
    def test_sixty_four_times_eight_states_counts_of_eight(self):
        k = 8
        self.app.register_model(
            "eight", tuple(str(i) for i in range(k)), ("x",), (0.125,) * k,
            tuple(tuple(1.0 / k for _ in range(k)) for _ in range(k)),
            ((1.0,),) * k)
        self.app.archive_recording("eight", (None,) * 64)
        conditions = (
            CountCondition(0, 32, "7", 8),
            CountCondition(16, 64, "0", 8),
        )
        windows = (
            Window("first", 0, 64, ("7", "0")),
            Window("overlap", 16, 48, ("0", "7")),
            Window("empty", 60, 60, ("1", "2")),
        )
        events = []
        key = self.app.analyze_counts("eight", "eight", conditions, windows, events.append)
        result = self.app.get_analysis(key)
        self.assertEqual(len(events), 64)
        self.assertTrue(math.isfinite(result.log_likelihood))
        self.assertTrue(math.isfinite(result.log_event_probability))
        self.assertLessEqual(result.log_event_probability, 0.0)
        for event in events:
            self.assertTrue(math.isclose(math.fsum(event.probabilities), 1.0,
                                         abs_tol=1e-12))
            self.assertTrue(all(0.0 <= p <= 1.0 for p in event.probabilities))
        # Independent probability-space check of the event probability, without
        # enumerating the 8^64 path space.
        width = 9
        alpha = [[0.0] * width for _ in range(width)]
        alpha[0][0] = 1.0
        for t in range(64):
            updated = [[0.0] * width for _ in range(width)]
            for c1 in range(width):
                for c2 in range(width):
                    mass = alpha[c1][c2]
                    if not mass:
                        continue
                    hit1 = t < 32
                    hit2 = t >= 16
                    for state in range(k):
                        n1 = c1 + (1 if hit1 and state == 7 else 0)
                        n2 = c2 + (1 if hit2 and state == 0 else 0)
                        if n1 < width and n2 < width:
                            updated[n1][n2] += mass / k
            alpha = updated
        self.assertTrue(math.isclose(math.exp(result.log_event_probability),
                                     alpha[8][8], rel_tol=1e-9))
        full, overlap, empty = result.reports
        self.assertEqual(empty.mean, (0.0, 0.0))
        self.assertEqual(empty.covariance, ((0.0, 0.0), (0.0, 0.0)))
        for report in (full, overlap):
            for a in range(2):
                self.assertTrue(0.0 <= report.mean[a] <= report.window.stop - report.window.start)
                self.assertGreaterEqual(report.covariance[a][a], 0.0)
                self.assertTrue(math.isclose(
                    report.covariance[0][1], report.covariance[1][0], abs_tol=1e-12))
