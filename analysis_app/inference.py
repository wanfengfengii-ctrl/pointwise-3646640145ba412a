"""Log-domain inference with explicitly materialized two-slice joint posteriors."""
import math
from dataclasses import dataclass


class ImpossibleSequence(ValueError):
    """The entire recording has zero likelihood under the selected model."""


NEG_INF = -math.inf


def log(value):
    return math.log(value) if value else NEG_INF


def logsum(values):
    values = tuple(values)
    largest = max(values, default=NEG_INF)
    if largest == NEG_INF:
        return NEG_INF
    return largest + math.log(math.fsum(math.exp(x - largest) for x in values))


def probabilities(log_values):
    total = logsum(log_values)
    if total == NEG_INF:
        return tuple(0.0 for _ in log_values)
    return tuple(math.exp(x - total) for x in log_values)


@dataclass(frozen=True)
class Posterior:
    marginals: tuple
    joints: tuple
    log_likelihood: float


def materialize(model, observations):
    """Materialized reference implementation used by the normal production path."""
    n, k = len(observations), len(model.states)
    symbol_indices = {name: i for i, name in enumerate(model.symbols)}
    if any(value is not None and value not in symbol_indices for value in observations):
        raise ValueError("recording contains a symbol outside the model alphabet")
    if not n:
        return Posterior((), (), 0.0)
    transition = tuple(tuple(log(x) for x in row) for row in model.transitions)
    emissions = []
    for value in observations:
        emissions.append(
            (0.0,) * k if value is None else
            tuple(log(model.emissions[s][symbol_indices[value]]) for s in range(k))
        )
    forward, scales = [], []
    raw = [log(model.initial[s]) + emissions[0][s] for s in range(k)]
    for t in range(n):
        if t:
            raw = [emissions[t][j] + logsum(
                forward[-1][i] + transition[i][j] for i in range(k)
            ) for j in range(k)]
        scale = logsum(raw)
        if scale == NEG_INF:
            raise ImpossibleSequence("recording has zero likelihood")
        scales.append(scale)
        forward.append(tuple(value - scale for value in raw))
    backward = [None] * n
    backward[-1] = (0.0,) * k
    for t in range(n - 2, -1, -1):
        raw = [logsum(transition[i][j] + emissions[t + 1][j] + backward[t + 1][j]
                      for j in range(k)) for i in range(k)]
        shift = max(raw)
        backward[t] = tuple(x - shift for x in raw)
    marginals = tuple(probabilities(tuple(forward[t][s] + backward[t][s]
                                           for s in range(k))) for t in range(n))
    joints = []
    for t in range(n - 1):
        values = tuple(forward[t][i] + transition[i][j] + emissions[t + 1][j]
                       + backward[t + 1][j] for i in range(k) for j in range(k))
        flat = probabilities(values)
        joints.append(tuple(tuple(flat[i * k + j] for j in range(k)) for i in range(k)))
    return Posterior(marginals, tuple(joints), math.fsum(scales))


def infer(model, observations):
    return materialize(model, observations)
