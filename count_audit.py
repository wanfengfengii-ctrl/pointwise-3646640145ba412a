"""Independent short-path audit for count-event conditioning.

Runs outside the unittest package: enumerates every state path with exact
Fraction arithmetic and checks the production entry point item by item on joint
mass, event probability, chronological evidence and full window moments.
Also exercises receiver failure, database publication failure, preservation of
older reports and reopen-from-file behaviour.  Exit status is non-zero on any
mismatch; success prints the audit summary.
"""
import math
import sqlite3
import sys
import tempfile
from fractions import Fraction as F
from pathlib import Path

from analysis_app import App, Window, CountCondition, CountAnalysis, ImpossibleSequence
from tests.count_oracle import exact_counts

TOL = 2e-10


def close(actual, expected):
    return math.isclose(actual, float(expected), abs_tol=TOL, rel_tol=TOL)


def frac_log(value):
    return -math.inf if value == 0 else math.log(value.numerator) - math.log(value.denominator)


def build(app, tag, initial, transition, emissions, observed, symbols=("a", "b")):
    k = len(initial)
    names = tuple("s" + str(i) for i in range(k))
    app.register_model(tag, names, symbols, tuple(map(float, initial)),
                       tuple(tuple(map(float, row)) for row in transition),
                       tuple(tuple(map(float, row)) for row in emissions))
    app.archive_recording(tag, tuple(None if x is None else symbols[x] if isinstance(x, int)
                                     else x for x in observed))
    return names


def audit_case(label, initial, transition, emissions, observed, condition_specs,
               window_specs, expect_zero=False, symbols=("a", "b")):
    with App(":memory:") as app:
        names = build(app, label, initial, transition, emissions, observed, symbols)
        conditions = tuple(CountCondition(start, stop, names[state], count)
                           for start, stop, state, count in condition_specs)
        windows = tuple(Window(str(i), start, stop, (names[a], names[b]))
                        for i, (start, stop, (a, b)) in enumerate(window_specs))
        events = []
        try:
            joint, plain, expected_marginals, expected_moments = exact_counts(
                initial, transition, emissions, observed, condition_specs,
                [(start, stop, targets) for start, stop, targets in window_specs])
        except ValueError:
            assert expect_zero, label + ": oracle saw zero mass unexpectedly"
            try:
                app.analyze_counts(label, label, conditions, windows, events.append)
            except ImpossibleSequence:
                assert events == [], label + ": evidence delivered on zero event"
                return label + ": zero mass rejected, no evidence"
            raise AssertionError(label + ": production did not reject zero event mass")
        assert not expect_zero, label + ": expected zero mass but oracle found paths"
        key = app.analyze_counts(label, label, conditions, windows, events.append)
        result = app.get_analysis(key)
        assert isinstance(result, CountAnalysis), label
        assert close(result.log_likelihood, frac_log(joint)), label + ": log P(O,E)"
        assert close(result.log_event_probability, frac_log(joint / plain)), label + ": log P(E|O)"
        assert len(events) == len(observed), label + ": evidence count"
        assert [e.time for e in events] == list(range(len(observed))), label + ": evidence order"
        for event, wanted in zip(events, expected_marginals):
            for value, rational in zip(event.probabilities, wanted):
                assert close(value, rational), (label, "marginal", value, rational)
        assert len(result.reports) == len(window_specs), label
        for report, (mean, covariance) in zip(result.reports, expected_moments):
            for a in range(2):
                assert close(report.mean[a], mean[a]), (label, "mean", report.mean, mean)
                for b in range(2):
                    assert close(report.covariance[a][b], covariance[a][b]), \
                        (label, "covariance", report.covariance, covariance)
        return label + ": joint mass, evidence and full moments match"


def audit_rare_underflow():
    # Tiny but positive joint mass: its probability underflows, logs must not.
    # Observing symbol "a" twice: an all-"off" event path carries a tiny factor
    # at each time, so the joint probability underflows while its log is finite.
    epsilon = 1e-162
    with App(":memory:") as app:
        app.register_model("rare", ("on", "off"), ("a", "b"), (0.5, 0.5),
                           ((0.5, 0.5), (0.5, 0.5)),
                           ((1.0 - epsilon, epsilon), (epsilon, 1.0 - epsilon)))
        app.archive_recording("rare", ("a", "a"))
        conditions = (CountCondition(0, 2, "off", 2),)
        key = app.analyze_counts("rare", "rare", conditions,
                                 (Window("w", 0, 2, ("on", "off")),), lambda event: None)
        result = app.get_analysis(key)
        assert math.isfinite(result.log_likelihood) and math.isfinite(result.log_event_probability)
        assert math.exp(result.log_likelihood) == 0.0, "tiny mass must underflow as a probability"
        registered = app.models.get("rare").emissions[1][0]
        assert registered > 0.0, "the tiny factor must remain a positive structural value"
        # Event path weight: initial 1/2 * off->off 1/2 * epsilon * epsilon.
        expected_joint = math.log(0.25) + 2 * math.log(registered)
        assert close(result.log_likelihood, expected_joint)
        # The four two-time path emission factors telescope to (1-e)^2 +
        # 2e(1-e) + e^2 = 1, so plain P(O) = 0.25 and P(E|O) = epsilon^2.
        assert close(result.log_event_probability, 2 * math.log(registered))
        assert result.reports[0].mean == (0.0, 2.0)
        return "rare: positive sub-float mass kept as finite logarithm"


def audit_boundaries():
    with tempfile.TemporaryDirectory(prefix="count-audit-") as directory:
        path = Path(directory) / "archive.sqlite"
        with App(path) as app:
            app.register_model("m", ("off", "on"), ("x",), (0.5, 0.5),
                               ((0.5, 0.5), (0.5, 0.5)), ((1.0,), (1.0,)))
            app.archive_recording("r", (None,) * 4)
            conditions = (CountCondition(0, 4, "on", 1),)
            windows = (Window("w", 0, 4, ("on", "off")),)

            # Receiver failure: prefix delivered, nothing published, retry keeps ID.
            seen = []
            marker = RuntimeError("receiver disconnected")
            def failing_sink(event):
                seen.append(event.time)
                if event.time == 1:
                    raise marker
            try:
                app.analyze_counts("r", "m", conditions, windows, failing_sink)
            except RuntimeError as exc:
                assert exc is marker
            assert seen == [0, 1]
            assert app.store.connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0
            key = app.analyze_counts("r", "m", conditions, windows, lambda event: None)
            assert key == "analysis-1"
            original = app.get_analysis(key)

            # Database publication failure rolls back and preserves the old report.
            app.store.connection.execute("""
                CREATE TRIGGER reject_publication BEFORE INSERT ON analyses
                BEGIN SELECT RAISE(ABORT,'publication storage rejected insert'); END
            """)
            try:
                app.analyze_counts("r", "m", (CountCondition(0, 4, "on", 2),),
                                   windows, lambda event: None)
            except sqlite3.IntegrityError:
                pass
            assert app.get_analysis("analysis-1") == original
            app.store.connection.execute("DROP TRIGGER reject_publication")
            second = app.analyze_counts("r", "m", (CountCondition(0, 4, "on", 2),),
                                        windows, lambda event: None)
            assert second == "analysis-2"
        with App(path) as app:
            reopened = app.get_analysis(key)
            assert reopened == original and isinstance(reopened, CountAnalysis)
            assert reopened.conditions == conditions
        return "boundaries: receiver abort, storage abort, old report, reopen all verified"


def main():
    fair2 = ((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
    initial2 = (F(1, 2), F(1, 2))
    emit_flat = ((F(1),), (F(1),))
    notes = []

    # Overlapping conditions on four fully missing times (the overlap hand case).
    notes.append(audit_case(
        "overlap", initial2, fair2, emit_flat, (None,) * 4,
        ((0, 3, 1, 1), (1, 4, 1, 1)),
        ((0, 4, (1, 0)), (0, 2, (0, 1)), (1, 3, (1, 0))),
        symbols=("x",)))

    # Duplicate conditions must not multiply event probability.
    notes.append(audit_case(
        "duplicate", initial2, fair2, emit_flat, (None,) * 3,
        ((0, 3, 1, 1), (0, 3, 1, 1)),
        ((0, 3, (1, 0)),), symbols=("x",)))

    # Contradictory legal conditions: zero joint mass, not a ValueError.
    notes.append(audit_case(
        "conflict", initial2, fair2, emit_flat, (None,) * 3,
        ((0, 3, 1, 0), (0, 3, 1, 1)), (), expect_zero=True, symbols=("x",)))

    # Structural zeros plus a zero-count prohibition (observations are symbol ids).
    locked = ((F(1), F(0)), (F(0), F(1)))
    notes.append(audit_case(
        "structural", (F(1), F(0)), locked, ((F(1), F(0)), (F(0), F(1))),
        (0, None, None, 0), ((0, 4, 1, 0),),
        ((0, 4, (0, 1)), (1, 3, (1, 0)))))

    # Tail hard evidence constraining earlier windows, with non-degenerate dynamics.
    notes.append(audit_case(
        "tail", initial2, ((F(4, 5), F(1, 5)), (F(3, 10), F(7, 10))),
        ((F(1), F(0)), (F(0), F(1))), (None, None, None, 1),
        ((0, 4, 1, 2),), ((0, 2, (0, 1)), (1, 4, (1, 0)))))

    # Empty recording and empty window.
    notes.append(audit_case(
        "empty", initial2, fair2, emit_flat, (), ((0, 0, 1, 0),),
        ((0, 0, (0, 1)),), symbols=("x",)))

    # Three-state chain with mixed hard/missing observations.
    r3 = ((F(2, 5), F(2, 5), F(1, 5)),
          (F(1, 5), F(3, 5), F(1, 5)),
          (F(1, 10), F(3, 10), F(3, 5)))
    initial3 = (F(1, 3), F(1, 3), F(1, 3))
    emit3 = ((F(1, 2), F(1, 2)), (F(1, 4), F(3, 4)), (F(2, 3), F(1, 3)))
    notes.append(audit_case(
        "three-state", initial3, r3, emit3, (None, 0, 1, None, 0),
        ((0, 5, 2, 1), (2, 5, 0, 1)),
        ((0, 5, (2, 0)), (1, 4, (0, 2)), (3, 3, (1, 2)))))

    notes.append(audit_rare_underflow())
    notes.append(audit_boundaries())
    for note in notes:
        print("audit:", note)
    print("count audit passed:", len(notes), "scenarios")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - audit entry reports any failure
        print("count audit FAILED:", type(exc).__name__, exc, file=sys.stderr)
        raise
