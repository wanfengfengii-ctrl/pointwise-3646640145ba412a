"""Independent exact-rational enumeration for count-event conditioning.

Every full state path is enumerated with Fraction weights; paths not satisfying
the intersection of the count conditions are discarded.  The oracle does not
import the production count inference or moment code.
"""
from fractions import Fraction
from itertools import product


def exact_counts(initial, transition, emissions, observations, conditions, windows):
    """Return (log values, marginals, moments) computed by full enumeration.

    conditions: tuples (start, stop, state_index, count); the event is their
    intersection and each matching time contributes to every applicable condition.
    windows: tuples (start, stop, (target_a, target_b)).
    """
    k, n = len(initial), len(observations)
    observations = tuple(None if x is None else int(x) for x in observations)
    weighted = []
    for path in product(range(k), repeat=n):
        weight = Fraction(1) if not n else initial[path[0]]
        for t, state in enumerate(path):
            if t:
                weight *= transition[path[t - 1]][state]
            if observations[t] is not None:
                weight *= emissions[state][observations[t]]
        if not weight:
            continue
        if all(sum(path[t] == state for t in range(start, stop)) == count
               for start, stop, state, count in conditions):
            weighted.append((path, weight))
    joint = sum((w for _, w in weighted), Fraction())
    if not joint:
        raise ValueError("zero event mass")
    # Plain observation mass requires enumerating all paths again, independently
    # of the event filtering above.
    plain = Fraction()
    for path in product(range(k), repeat=n):
        weight = Fraction(1) if not n else initial[path[0]]
        for t, state in enumerate(path):
            if t:
                weight *= transition[path[t - 1]][state]
            if observations[t] is not None:
                weight *= emissions[state][observations[t]]
        plain += weight
    if not plain:
        raise ValueError("zero likelihood")
    posterior = [[Fraction() for _ in range(k)] for _ in range(n)]
    for path, weight in weighted:
        p = weight / joint
        for t, state in enumerate(path):
            posterior[t][state] += p
    moments = []
    for start, stop, targets in windows:
        mean = [Fraction(), Fraction()]
        second = [[Fraction(), Fraction()], [Fraction(), Fraction()]]
        for path, weight in weighted:
            p = weight / joint
            values = [sum(path[t] == state for t in range(start, stop)) for state in targets]
            for a in range(2):
                mean[a] += p * values[a]
                for b in range(2):
                    second[a][b] += p * values[a] * values[b]
        covariance = [[second[a][b] - mean[a] * mean[b] for b in range(2)] for a in range(2)]
        moments.append((mean, covariance))
    return joint, plain, posterior, moments
