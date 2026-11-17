"""Count-event conditioning over an augmented (state, counter) first-order chain.

The event E is the intersection of one or two occupation-count conditions.  The
augmented process Y_t = (X_t, c_1(t), c_2(t)) is first-order Markov: every counter
increment is a deterministic function of time and the current state.  Running the
forward-backward recursion on this augmented chain in the log domain yields the
event-conditioned posterior P(Y_t | O, E) without ever projecting the event back
onto pairwise state marginals.  Positive joint mass stays finite as a logarithm
even when its probability would underflow.
"""
import math
from dataclasses import dataclass
from .inference import NEG_INF, log, logsum, probabilities, materialize, ImpossibleSequence
from .occupancy import Occupancy


@dataclass(frozen=True)
class CountCondition:
    start: int
    stop: int
    state: str
    count: int


@dataclass(frozen=True)
class CountAnalysis:
    analysis_id: str
    recording_id: str
    model_id: str
    states: tuple
    length: int
    log_likelihood: float
    reports: tuple
    conditions: tuple
    log_event_probability: float


@dataclass(frozen=True)
class _CountedPosterior:
    marginals: tuple
    log_joint: float
    log_observation: float
    log_event_probability: float
    alpha: tuple
    beta: tuple
    transitions: tuple
    emissions: tuple
    dimensions: tuple
    specs: tuple
    state_count: int
    length: int


def validate_conditions(conditions, n, state_names):
    """Snapshot and qualify count conditions; all failures are ValueError."""
    try:
        snapshot = tuple(conditions)
    except TypeError as exc:
        raise ValueError("conditions must be iterable") from exc
    if not 1 <= len(snapshot) <= 2:
        raise ValueError("analyze_counts expects one or two CountCondition items")
    qualified = []
    for item in snapshot:
        if not isinstance(item, CountCondition):
            raise ValueError("each condition must be a CountCondition")
        if type(item.start) is not int or type(item.stop) is not int or type(item.count) is not int:
            raise ValueError("condition bounds and counts must be non-boolean integers")
        if not isinstance(item.state, str) or item.state not in state_names:
            raise ValueError("condition state must be a registered model state")
        if not 0 <= item.start <= item.stop <= n:
            raise ValueError("condition interval is outside the recording")
        if not 0 <= item.count <= min(8, item.stop - item.start):
            raise ValueError("condition count is outside 0..min(8, interval length)")
        qualified.append(CountCondition(item.start, item.stop, item.state, item.count))
    return tuple(qualified)


def _index(state, c1, c2, d1, d2):
    return (state * d1 + c1) * d2 + c2


def _decode(index, d1, d2):
    c2 = index % d2
    state_c1 = index // d2
    return state_c1 // d1, state_c1 % d1, c2


def constrain(model, observations, conditions):
    """Log-domain constrained forward-backward; raises on zero joint mass."""
    # The ordinary pass also validates the alphabet and the plain zero-likelihood
    # boundary before any event-conditioned evidence is produced.
    ordinary = materialize(model, observations)
    n, k = len(observations), len(model.states)
    specs = tuple((c.start, c.stop, model.states.index(c.state), c.count) for c in conditions)
    two = len(specs) == 2
    d1 = specs[0][3] + 1
    d2 = specs[1][3] + 1 if two else 1
    targets = (specs[0][3], specs[1][3] if two else 0)
    if n == 0:
        return _CountedPosterior((), 0.0, ordinary.log_likelihood, 0.0, (), (),
                                 (), (), (d1, d2), specs, k, n)
    size = k * d1 * d2
    transitions = tuple(tuple(log(x) for x in row) for row in model.transitions)
    symbol_indices = {name: i for i, name in enumerate(model.symbols)}
    emissions = tuple(
        (0.0,) * k if value is None else
        tuple(log(model.emissions[s][symbol_indices[value]]) for s in range(k))
        for value in observations)
    initial = tuple(log(x) for x in model.initial)

    def increment(time, state, dim):
        start, stop, condition_state, _ = specs[dim]
        return 1 if start <= time < stop and state == condition_state else 0

    alpha = [None] * n
    table = [NEG_INF] * size
    for state in range(k):
        c1 = increment(0, state, 0)
        c2 = increment(0, state, 1) if two else 0
        if c1 < d1 and c2 < d2:
            table[_index(state, c1, c2, d1, d2)] = initial[state] + emissions[0][state]
    alpha[0] = tuple(table)
    for time in range(1, n):
        table = [NEG_INF] * size
        for u, value in enumerate(alpha[time - 1]):
            if value == NEG_INF:
                continue
            state, c1, c2 = _decode(u, d1, d2)
            for nxt in range(k):
                e1 = c1 + increment(time, nxt, 0)
                e2 = c2 + increment(time, nxt, 1) if two else c2
                if e1 < d1 and e2 < d2:
                    v = _index(nxt, e1, e2, d1, d2)
                    candidate = value + transitions[state][nxt] + emissions[time][nxt]
                    table[v] = candidate if table[v] == NEG_INF else logsum((table[v], candidate))
        alpha[time] = tuple(table)

    beta = [None] * n
    table = [NEG_INF] * size
    for u in range(size):
        _, c1, c2 = _decode(u, d1, d2)
        if c1 == targets[0] and c2 == targets[1]:
            table[u] = 0.0
    beta[n - 1] = tuple(table)
    for time in range(n - 1, 0, -1):
        table = [NEG_INF] * size
        for u in range(size):
            state, c1, c2 = _decode(u, d1, d2)
            for nxt in range(k):
                e1 = c1 + increment(time, nxt, 0)
                e2 = c2 + increment(time, nxt, 1) if two else c2
                if e1 < d1 and e2 < d2:
                    v = _index(nxt, e1, e2, d1, d2)
                    tail = beta[time][v]
                    if tail != NEG_INF:
                        candidate = transitions[state][nxt] + emissions[time][nxt] + tail
                        table[u] = candidate if table[u] == NEG_INF else logsum((table[u], candidate))
        beta[time - 1] = tuple(table)

    final_mass = []
    for u in range(size):
        _, c1, c2 = _decode(u, d1, d2)
        if c1 == targets[0] and c2 == targets[1]:
            final_mass.append(alpha[n - 1][u])
    log_joint = logsum(final_mass)
    if log_joint == NEG_INF:
        raise ImpossibleSequence("count conditions have zero joint mass")
    marginals = []
    stride = d1 * d2
    for time in range(n):
        state_logs = tuple(
            logsum(alpha[time][u] + beta[time][u]
                   for u in range(state * stride, (state + 1) * stride))
            for state in range(k))
        marginals.append(probabilities(state_logs))
    return _CountedPosterior(
        tuple(marginals), log_joint, ordinary.log_likelihood,
        log_joint - ordinary.log_likelihood,
        tuple(alpha), tuple(beta), transitions, tuple(emissions),
        (d1, d2), specs, k, n)


def counted_reports(counted, windows, state_names):
    """Window occupation moments propagated on the event-conditioned augmented chain."""
    k = counted.state_count
    d1, d2 = counted.dimensions
    stride = d1 * d2
    size = k * stride
    alpha, beta = counted.alpha, counted.beta
    transitions, emissions = counted.transitions, counted.emissions
    specs = counted.specs
    two = len(specs) == 2

    def increment(time, state, dim):
        start, stop, condition_state, _ = specs[dim]
        return 1 if start <= time < stop and state == condition_state else 0

    reports = []
    for window in windows:
        if window.start == window.stop:
            reports.append(Occupancy(window, (0.0, 0.0), ((0.0, 0.0), (0.0, 0.0))))
            continue
        targets = tuple(state_names.index(name) for name in window.states)
        t0 = window.start
        mass = [0.0] * size
        first = [[0.0, 0.0] for _ in range(size)]
        second = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(size)]
        for u in range(size):
            value = alpha[t0][u] + beta[t0][u]
            if value == NEG_INF:
                continue
            state = u // stride
            probability = math.exp(value - counted.log_joint)
            mass[u] = probability
            for a in range(2):
                first[u][a] = probability * (state == targets[a])
                for b in range(2):
                    second[u][a][b] = probability * (state == targets[a]) * (state == targets[b])
        for time in range(t0, window.stop - 1):
            new_mass = [0.0] * size
            new_first = [[0.0, 0.0] for _ in range(size)]
            new_second = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(size)]
            for u in range(size):
                if not mass[u]:
                    continue
                state, c1, c2 = _decode(u, d1, d2)
                weights = []
                for nxt in range(k):
                    e1 = c1 + increment(time + 1, nxt, 0)
                    e2 = c2 + increment(time + 1, nxt, 1) if two else c2
                    if e1 < d1 and e2 < d2:
                        v = _index(nxt, e1, e2, d1, d2)
                        tail = beta[time + 1][v]
                        if tail != NEG_INF:
                            log_weight = (transitions[state][nxt] + emissions[time + 1][nxt]
                                          + tail - beta[time][u])
                            weights.append((nxt, v, math.exp(log_weight)))
                normalizer = math.fsum(weight for _, _, weight in weights)
                for nxt, v, raw in weights:
                    # q is P(Y_t=u, Y_{t+1}=v | O, E); the joint mass already
                    # carries every cross-time dependency introduced by the event.
                    q = raw / normalizer * mass[u]
                    step = (int(nxt == targets[0]), int(nxt == targets[1]))
                    new_mass[v] += q
                    for a in range(2):
                        new_first[v][a] += q * (first[u][a] / mass[u] + step[a])
                        for b in range(2):
                            new_second[v][a][b] += q * (
                                second[u][a][b] / mass[u]
                                + step[a] * first[u][b] / mass[u]
                                + step[b] * first[u][a] / mass[u]
                                + step[a] * step[b])
            mass, first, second = new_mass, new_first, new_second
        mean = tuple(math.fsum(first[u][a] for u in range(size)) for a in range(2))
        raw_second = tuple(tuple(
            math.fsum(second[u][a][b] for u in range(size)) for b in range(2))
            for a in range(2))
        covariance = tuple(tuple(
            raw_second[a][b] - mean[a] * mean[b] for b in range(2)) for a in range(2))
        # Only roundoff-sized residuals are clipped: tiny negative diagonals to
        # zero, and antisymmetric off-diagonal noise is averaged away so the
        # published covariance stays symmetric without hiding real dependencies.
        scale = max(1.0, *(mean[a] ** 2 for a in range(2)))
        cleaned = [[0.0, 0.0], [0.0, 0.0]]
        for a in range(2):
            for b in range(2):
                value = covariance[a][b]
                if a == b and value < 0.0 and value > -1e-12 * scale:
                    value = 0.0
                cleaned[a][b] = value
        if abs(cleaned[0][1] - cleaned[1][0]) <= 1e-12 * scale:
            shared = 0.5 * (cleaned[0][1] + cleaned[1][0])
            cleaned[0][1] = cleaned[1][0] = shared
        covariance = tuple(tuple(row) for row in cleaned)
        reports.append(Occupancy(window, mean, covariance))
    return tuple(reports)
