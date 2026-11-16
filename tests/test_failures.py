import sqlite3
from analysis_app import Window
from tests.helpers import Case


class Failures(Case):
    def test_sink_exception_identity_and_successful_retry(self):
        self.persistent((None,) * 4)
        marker = RuntimeError("receiver disconnected")
        seen = []
        def receiver(event):
            seen.append(event.time)
            if event.time == 1:
                raise marker
        with self.assertRaises(RuntimeError) as raised:
            self.app.analyze("r", "m", [Window("w", 0, 4, ("off", "on"))], receiver)
        self.assertIs(raised.exception, marker)
        self.assertEqual(seen, [0, 1])
        self.no_publication()
        result, events = self.analyze()
        self.assertEqual(result.analysis_id, "analysis-1")
        self.assertEqual(len(events), 4)

    def test_publication_database_abort_keeps_old_result(self):
        self.persistent()
        old, _ = self.analyze()
        connection = self.app.store.connection
        connection.execute("""
            CREATE TRIGGER reject_publication BEFORE INSERT ON analyses
            BEGIN SELECT RAISE(ABORT,'publication storage rejected insert'); END
        """)
        seen = []
        with self.assertRaises(sqlite3.IntegrityError):
            self.app.analyze("r", "m", (), seen.append)
        self.assertEqual([e.time for e in seen], [0, 1])
        self.assertEqual(self.app.get_analysis(old.analysis_id), old)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0], 1)
        self.assertFalse(connection.in_transaction)
        with self.assertRaises(KeyError):
            self.app.get_analysis("analysis-2")
        connection.execute("DROP TRIGGER reject_publication")
        new, _ = self.analyze(())
        self.assertEqual(new.analysis_id, "analysis-2")
        self.assertEqual(new.reports, ())

    def test_source_exception_has_no_partial_archive(self):
        marker = OSError("source interrupted")
        def source():
            yield "x"
            raise marker
        with self.assertRaises(OSError) as raised:
            self.app.archive_recording("partial", source())
        self.assertIs(raised.exception, marker)
        with self.assertRaises(KeyError):
            self.app.recordings.get("partial")
        self.assertFalse(self.app.store.connection.in_transaction)
