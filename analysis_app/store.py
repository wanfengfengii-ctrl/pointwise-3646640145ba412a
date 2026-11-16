"""Single-worker SQLite storage. Each public write is atomic."""
import json
import sqlite3
from contextlib import contextmanager


def identifier(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("expected a nonempty string without NUL")
    return value


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Store:
    def __init__(self, path):
        self.connection = sqlite3.connect(path, isolation_level=None)
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS recordings(
                    id TEXT PRIMARY KEY, observations TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS models(
                    id TEXT PRIMARY KEY, definition TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS analyses(
                    number INTEGER PRIMARY KEY,
                    recording_id TEXT NOT NULL REFERENCES recordings(id),
                    model_id TEXT NOT NULL REFERENCES models(id),
                    payload TEXT NOT NULL);
                PRAGMA user_version=1;
            """)
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def close(self):
        self.connection.close()
