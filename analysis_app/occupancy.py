"""Window occupation moments under a full-sequence conditional posterior chain."""
import math
from dataclasses import dataclass
from .store import identifier


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    stop: int
    states: tuple


@dataclass(frozen=True)
class Occupancy:
    window: Window
    mean: tuple
    covariance: tuple


def validate_windows(windows, n, state_names):
    try:
        windows = tuple(windows)
    except TypeError as exc:
        raise ValueError("windows must be iterable") from exc
    if len(windows) > 16:
        raise ValueError("at most 16 windows")
    result, seen = [], set()
    for window in windows:
        if not isinstance(window, Window):
            raise ValueError("expected Window")
        identifier(window.name)
        if window.name in seen:
            raise ValueError("duplicate window name")
        seen.add(window.name)
        if (type(window.start) is not int or type(window.stop) is not int
                or not 0 <= window.start <= window.stop <= n):
            raise ValueError("invalid half-open window")
        try:
            states = tuple(window.states)
        except TypeError as exc:
            raise ValueError("expected two distinct state names") from exc
        if (len(states) != 2 or any(not isinstance(s, str) for s in states)
                or states[0] == states[1] or any(s not in state_names for s in states)):
            raise ValueError("expected two distinct known state names")
        result.append(Window(window.name, window.start, window.stop, states))
    return tuple(result)


def materialized_reports(posterior, windows, state_names):
    """Reference raw first/second moment propagation. Copies joint tables per report."""
    k = len(state_names)
    reports = []
    for window in windows:
        if window.start == window.stop:
            reports.append(Occupancy(window, (0.0, 0.0), ((0.0, 0.0), (0.0, 0.0))))
            continue
        # Report preparation owns a separate snapshot of the complete two-slice evidence.
        joints = [[list(row) for row in table] for table in posterior.joints]
        marginals = [list(row) for row in posterior.marginals]
        targets = tuple(state_names.index(s) for s in window.states)
        start, stop = window.start, window.stop
        mass = marginals[start]
        first = [[mass[s] * (s == targets[a]) for a in range(2)] for s in range(k)]
        second = [[[mass[s] * (s == targets[a]) * (s == targets[b])
                    for b in range(2)] for a in range(2)] for s in range(k)]
        for t in range(start, stop - 1):
            new_mass = [0.0] * k
            new_first = [[0.0, 0.0] for _ in range(k)]
            new_second = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(k)]
            for i in range(k):
                row_total = math.fsum(joints[t][i])
                if not row_total:
                    continue
                for j in range(k):
                    p = joints[t][i][j] / row_total
                    if not p:
                        continue
                    increment = (int(j == targets[0]), int(j == targets[1]))
                    new_mass[j] += p * mass[i]
                    for a in range(2):
                        new_first[j][a] += p * (first[i][a] + increment[a] * mass[i])
                        for b in range(2):
                            new_second[j][a][b] += p * (
                                second[i][a][b] + increment[a] * first[i][b]
                                + increment[b] * first[i][a]
                                + increment[a] * increment[b] * mass[i])
            mass, first, second = new_mass, new_first, new_second
        total = math.fsum(mass)
        mean = tuple(math.fsum(first[s][a] for s in range(k)) / total for a in range(2))
        raw_second = tuple(tuple(math.fsum(second[s][a][b] for s in range(k)) / total
                                 for b in range(2)) for a in range(2))
        covariance = tuple(tuple(raw_second[a][b] - mean[a] * mean[b]
                                 for b in range(2)) for a in range(2))
        # Only roundoff-sized negative diagonal values are clipped.
        covariance = tuple(tuple(
            max(0.0, value) if a == b and value > -1e-12 * max(1.0, mean[a] ** 2) else value
            for b, value in enumerate(row)) for a, row in enumerate(covariance))
        reports.append(Occupancy(window, mean, covariance))
    return tuple(reports)


def summarize(posterior, windows, state_names):
    return materialized_reports(posterior, windows, state_names)
