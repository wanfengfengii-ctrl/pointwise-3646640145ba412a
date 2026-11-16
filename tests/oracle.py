"""Independent short-path enumeration using exact rational arithmetic."""
from fractions import Fraction
from itertools import product


def exact(initial, transition, emissions, observations, windows):
    k, n = len(initial), len(observations)
    weighted = []
    for path in product(range(k), repeat=n):
        weight = Fraction(1) if not n else initial[path[0]]
        for t, state in enumerate(path):
            if t:
                weight *= transition[path[t - 1]][state]
            if observations[t] is not None:
                weight *= emissions[state][observations[t]]
        if weight:
            weighted.append((path, weight))
    normalizer = sum((w for _, w in weighted), Fraction())
    if not normalizer:
        raise ValueError("zero likelihood")
    posterior = [[Fraction() for _ in range(k)] for _ in range(n)]
    counts = []
    for start, stop, targets in windows:
        mean = [Fraction(), Fraction()]
        second = [[Fraction(), Fraction()], [Fraction(), Fraction()]]
        for path, weight in weighted:
            p = weight / normalizer
            values = [sum(path[t] == state for t in range(start, stop)) for state in targets]
            for a in range(2):
                mean[a] += p * values[a]
                for b in range(2):
                    second[a][b] += p * values[a] * values[b]
        counts.append((mean, [[second[a][b] - mean[a] * mean[b] for b in range(2)] for a in range(2)]))
    for path, weight in weighted:
        for t, state in enumerate(path):
            posterior[t][state] += weight / normalizer
    return normalizer, posterior, counts
