import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from analysis_app import App, Window
from reference import ReferenceApp
from samples import load_demo
from tests.helpers import Case


class Publication(Case):
    def test_reanalysis_empty_window_and_old_result(self):
        self.persistent((None,) * 4)
        first, _ = self.analyze()
        second, events = self.analyze((Window("empty", 2, 2, ("off", "on")),))
        self.assertNotEqual(first.analysis_id, second.analysis_id)
        self.assertEqual(len(events), 4)
        self.assertEqual(second.reports[0].mean, (0.0, 0.0))
        self.assertEqual(second.reports[0].covariance, ((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual(self.app.get_analysis(first.analysis_id), first)
        with self.assertRaises(FrozenInstanceError):
            first.length = 10
        for key in ("missing", "analysis-0", "analysis-01", "analysis-1 OR 1=1",
                    "analysis-", "analysis-" + "9" * 50, None):
            with self.assertRaises(KeyError):
                self.app.get_analysis(key)

    def test_reopen_real_archive_and_failed_publication(self):
        with tempfile.TemporaryDirectory(prefix="state-analysis-test-") as directory:
            path = Path(directory) / "archive.sqlite"
            with App(path) as app:
                windows = load_demo(app)
                events = []
                key = app.analyze("shift", "equipment", windows, events.append)
                original = app.get_analysis(key)
                app.store.connection.execute("""
                    CREATE TRIGGER reject_publication BEFORE INSERT ON analyses
                    BEGIN SELECT RAISE(ABORT,'disk publication rejected'); END
                """)
                import sqlite3
                with self.assertRaises(sqlite3.IntegrityError):
                    app.analyze("shift", "equipment", windows, lambda event: None)
            with App(path) as app:
                self.assertEqual(app.get_analysis(key), original)
                self.assertEqual(len(app.recordings.get("shift")), 6)
                self.assertEqual(app.models.get("equipment").states, ("steady", "alarm"))
                with self.assertRaises(KeyError):
                    app.get_analysis("analysis-2")
                app.store.connection.execute("DROP TRIGGER reject_publication")
                new = app.analyze("shift", "equipment", windows, lambda event: None)
                self.assertEqual(new, "analysis-2")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["archive.sqlite"])

    def test_reference_publication_matches_production(self):
        with ReferenceApp(":memory:") as reference:
            expected_events, actual_events = [], []
            rw, aw = load_demo(reference), load_demo(self.app)
            rk = reference.analyze("shift", "equipment", rw, expected_events.append)
            ak = self.app.analyze("shift", "equipment", aw, actual_events.append)
            self.assertEqual(actual_events, expected_events)
            self.assertEqual(self.app.get_analysis(ak), reference.get_analysis(rk))
