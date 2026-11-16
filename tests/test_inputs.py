import math
from analysis_app import Window
from tests.helpers import Case


class Inputs(Case):
    def test_archive_validation_conflict_and_snapshot(self):
        values = ["a", None, "b"]
        self.app.archive_recording("r", values)
        values[0] = "changed"
        self.assertEqual(self.app.recordings.get("r"), ("a", None, "b"))
        self.assertEqual(self.app.archive_recording("r", ("a", None, "b")), "r")
        for invalid in (["a", 2], [""], ["a\x00b"], "abc", None):
            with self.assertRaises(ValueError):
                self.app.archive_recording("bad", invalid)
            with self.assertRaises(KeyError):
                self.app.recordings.get("bad")
        with self.assertRaises(ValueError):
            self.app.archive_recording("r", ("b",))
        self.assertEqual(self.app.recordings.get("r"), ("a", None, "b"))

    def test_model_validation_and_atomic_identity(self):
        good = (("off", "on"), ("x",), (0.5, 0.5),
                ((1.0, 0.0), (0.0, 1.0)), ((1.0,), (1.0,)))
        bads = [
            (("off", "off"),) + good[1:],
            (("off",),) + good[1:],
            good[:1] + (("x", "x"),) + good[2:],
            good[:2] + ((-0.1, 1.1),) + good[3:],
            good[:2] + ((math.nan, 0.5),) + good[3:],
            good[:2] + ((math.inf, 0.5),) + good[3:],
            good[:2] + ((True, False),) + good[3:],
            good[:2] + ((1e308, 1e308),) + good[3:],
            good[:2] + ((10 ** 500, 0),) + good[3:],
            good[:3] + (((0.2, 0.2), (0.0, 1.0)),) + good[4:],
            good[:4] + (((1.0,),),),
        ]
        for definition in bads:
            with self.assertRaises(ValueError):
                self.app.register_model("bad", *definition)
            with self.assertRaises(KeyError):
                self.app.models.get("bad")
        self.app.register_model("m", *good)
        before = self.app.models.get("m")
        with self.assertRaises(ValueError):
            self.app.register_model("m", *good)
        self.assertEqual(self.app.models.get("m"), before)
        self.assertFalse(self.app.store.connection.in_transaction)

    def test_bad_windows_and_unknown_inputs_deliver_nothing(self):
        self.persistent()
        bads = [
            [Window("w", -1, 2, ("off", "on"))],
            [Window("w", 2, 1, ("off", "on"))],
            [Window("w", 0, 3, ("off", "on"))],
            [Window("w", False, 1, ("off", "on"))],
            [Window("w", 0, 1, ("off", "off"))],
            [Window("w", 0, 1, ("off", "other"))],
            [Window("w", 0, 1, ("off", "on"))] * 2,
            [Window(str(i), 0, 1, ("off", "on")) for i in range(17)],
            [("w", 0, 1, ("off", "on"))],
        ]
        for windows in bads:
            events = []
            with self.assertRaises(ValueError):
                self.app.analyze("r", "m", windows, events.append)
            self.assertEqual(events, [])
            self.no_publication()
        for r, m in (("unknown", "m"), ("r", "unknown")):
            with self.assertRaises(KeyError):
                self.app.analyze(r, m, (), lambda event: None)
        with self.assertRaises(ValueError):
            self.app.analyze("r", "m", (), None)
        self.app.archive_recording("foreign", ("foreign",))
        events = []
        with self.assertRaises(ValueError):
            self.app.analyze("foreign", "m", (), events.append)
        self.assertEqual(events, [])
        self.no_publication()

    def test_identifiers_and_limit_windows(self):
        for value in ("", "\x00", 3, None):
            with self.assertRaises(ValueError):
                self.app.archive_recording(value, ())
        self.persistent()
        windows = [Window(str(i), 0, 2, ["off", "on"]) for i in range(16)]
        result, _ = self.analyze(windows)
        self.assertEqual(len(result.reports), 16)
        windows.clear()
        self.assertEqual(len(self.app.get_analysis(result.analysis_id).reports), 16)
