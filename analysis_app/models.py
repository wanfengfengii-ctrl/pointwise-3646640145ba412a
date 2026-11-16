"""Validated finite-state models with immutable state and symbol identities."""
import json
import math
from dataclasses import dataclass
from .store import identifier, encode


def names(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("expected a sequence of names")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("expected names") from exc
    for value in result:
        identifier(value)
    if len(set(result)) != len(result):
        raise ValueError("duplicate names")
    return result


def distribution(values, size):
    try:
        row = tuple(values)
    except TypeError as exc:
        raise ValueError("expected probability row") from exc
    if len(row) != size:
        raise ValueError("probability dimensions")
    if any(type(x) not in (int, float) or x < 0 or x > 1.0 + 1e-12 or not math.isfinite(x) for x in row):
        raise ValueError("probabilities must be finite and nonnegative")
    total = math.fsum(row)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("probability row must sum to one")
    return tuple(float(x) / total for x in row)


def matrix(values, rows, columns):
    try:
        result = tuple(distribution(row, columns) for row in values)
    except TypeError as exc:
        raise ValueError("expected probability matrix") from exc
    if len(result) != rows:
        raise ValueError("matrix dimensions")
    return result


@dataclass(frozen=True)
class Model:
    states: tuple
    symbols: tuple
    initial: tuple
    transitions: tuple
    emissions: tuple


class Models:
    def __init__(self, store):
        self.store = store

    def register_model(self, model_id, states, symbols, initial, transitions, emissions):
        identifier(model_id)
        states, symbols = names(states), names(symbols)
        k = len(states)
        if not 2 <= k <= 8 or not symbols:
            raise ValueError("expected 2..8 states and at least one symbol")
        definition = Model(states, symbols, distribution(initial, k),
                           matrix(transitions, k, k), matrix(emissions, k, len(symbols)))
        payload = encode([definition.states, definition.symbols, definition.initial,
                          definition.transitions, definition.emissions])
        with self.store.transaction():
            if self.store.connection.execute(
                "SELECT 1 FROM models WHERE id=?", (model_id,)
            ).fetchone():
                raise ValueError("model ID already registered")
            self.store.connection.execute("INSERT INTO models VALUES (?,?)", (model_id, payload))
        return model_id

    def get(self, model_id):
        identifier(model_id)
        row = self.store.connection.execute(
            "SELECT definition FROM models WHERE id=?", (model_id,)
        ).fetchone()
        if row is None:
            raise KeyError(model_id)
        states, symbols, initial, transitions, emissions = json.loads(row[0])
        return Model(tuple(states), tuple(symbols), tuple(initial),
                     tuple(map(tuple, transitions)), tuple(map(tuple, emissions)))
