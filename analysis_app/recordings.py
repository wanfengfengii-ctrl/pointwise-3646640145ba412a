"""Immutable ordered observation archives; None occupies a missing time slot."""
import json
from .store import identifier, encode


class Recordings:
    def __init__(self, store):
        self.store = store

    def archive_recording(self, recording_id, observations):
        identifier(recording_id)
        if isinstance(observations, (str, bytes)):
            raise ValueError("observations must be an iterable of symbols or None")
        try:
            values = tuple(observations)
        except TypeError as exc:
            raise ValueError("observations must be iterable") from exc
        for value in values:
            if value is not None:
                identifier(value)
        payload = encode(values)
        with self.store.transaction():
            old = self.store.connection.execute(
                "SELECT observations FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
            if old is not None:
                if json.loads(old[0]) != list(values):
                    raise ValueError("recording ID conflict")
            else:
                self.store.connection.execute(
                    "INSERT INTO recordings VALUES (?,?)", (recording_id, payload)
                )
        return recording_id

    def get(self, recording_id):
        identifier(recording_id)
        row = self.store.connection.execute(
            "SELECT observations FROM recordings WHERE id=?", (recording_id,)
        ).fetchone()
        if row is None:
            raise KeyError(recording_id)
        return tuple(json.loads(row[0]))
