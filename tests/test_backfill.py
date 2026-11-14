"""Unit tests for LibreView historical Recorder backfill logic."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = REPO_ROOT / "custom_components" / "libreview"


def _load_component_modules():
    """Load const.py and backfill.py as a package without requiring Home Assistant."""
    pkg_name = "custom_components_libreview_testpkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(COMPONENT_DIR)]
        sys.modules[pkg_name] = pkg

    const_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.const", COMPONENT_DIR / "const.py"
    )
    const_mod = importlib.util.module_from_spec(const_spec)
    sys.modules[f"{pkg_name}.const"] = const_mod
    const_spec.loader.exec_module(const_mod)

    backfill_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.backfill", COMPONENT_DIR / "backfill.py"
    )
    backfill_mod = importlib.util.module_from_spec(backfill_spec)
    sys.modules[f"{pkg_name}.backfill"] = backfill_mod
    backfill_spec.loader.exec_module(backfill_mod)
    return const_mod, backfill_mod


CONST, BACKFILL = _load_component_modules()


@dataclass
class FakeMeasurement:
    factory_timestamp: datetime
    value_in_mg_per_dl: int
    value: float
    trend_arrow: int | None = None


@dataclass
class FakeConnection:
    first_name: str = "Nicolas"
    last_name: str = "Navaroli"
    uom: int = 1  # 1 = mg/dL, 0 = mmol/L
    target_high: int = 180
    target_low: int = 70


class TestBackfillLogic(unittest.TestCase):
    """Validate pure backfill selection, chain repair, and statistics compilation."""

    def setUp(self) -> None:
        self.base_dt = datetime(2026, 11, 13, 7, 0, 0, tzinfo=timezone.utc)
        self.conn = FakeConnection()

    def _naive(self, offset_minutes: float) -> datetime:
        dt = self.base_dt + timedelta(minutes=offset_minutes)
        return dt.replace(tzinfo=None)

    def _ts(self, offset_minutes: float) -> float:
        return (self.base_dt + timedelta(minutes=offset_minutes)).timestamp()

    def test_build_backfill_points_mgdl_and_mmoll(self) -> None:
        raw = [
            FakeMeasurement(self._naive(30), 108, 108.0, trend_arrow=3),
            FakeMeasurement(self._naive(15), 126, 126.0, trend_arrow=2),
            FakeMeasurement(self._naive(15), 126, 126.0, trend_arrow=2),  # duplicate
        ]
        pts_mg = BACKFILL.build_backfill_points_from_graph(
            self.conn,
            raw,
            uom=CONST.GlucoseUnitOfMeasurement.MGDL,
            use_trend_icons=True,
        )
        self.assertEqual(len(pts_mg), 2)
        self.assertEqual(pts_mg[0].state_str, "126")
        self.assertAlmostEqual(pts_mg[0].value_mmol_l, 7.0)
        self.assertEqual(pts_mg[0].trend, "decreasing")
        self.assertEqual(pts_mg[0].icon, "mdi:arrow-bottom-right-thick")
        self.assertEqual(pts_mg[1].state_str, "108")

        pts_mmol = BACKFILL.build_backfill_points_from_graph(
            self.conn,
            raw,
            uom=CONST.GlucoseUnitOfMeasurement.MMOLL,
            use_trend_icons=False,
        )
        self.assertEqual(pts_mmol[0].state_str, "7.0")
        self.assertEqual(pts_mmol[0].icon, CONST.DEFAULT_ICON)

    def test_select_missing_points_overnight_gap_and_idempotency(self) -> None:
        # Simulate bedtime stopping at +25m (135 mg/dL), reopening app at +285m (104 mg/dL)
        existing = [
            BACKFILL.ExistingStateRecord(
                state_id=10,
                state="135",
                last_updated_ts=self._ts(25.3),
                last_changed_ts=None,
                old_state_id=9,
                measurement_ts=self._ts(25.1),
            ),
            # Unavailable transition recorded at +40m
            BACKFILL.ExistingStateRecord(
                state_id=11,
                state="unavailable",
                last_updated_ts=self._ts(40.0),
                last_changed_ts=None,
                old_state_id=10,
                measurement_ts=None,
            ),
            # Stale reading recorded if HA restarted at +120m while app still closed
            BACKFILL.ExistingStateRecord(
                state_id=12,
                state="135",
                last_updated_ts=self._ts(120.0),
                last_changed_ts=None,
                old_state_id=11,
                measurement_ts=self._ts(25.1),
            ),
            # Morning live reading at +285m
            BACKFILL.ExistingStateRecord(
                state_id=13,
                state="104",
                last_updated_ts=self._ts(285.3),
                last_changed_ts=None,
                old_state_id=12,
                measurement_ts=self._ts(285.1),
            ),
        ]

        # Graph points at +25m (already recorded live), +40m, +120m, +270m, +285m (live)
        graph_raw = [
            FakeMeasurement(self._naive(25.1), 135, 135.0),
            FakeMeasurement(self._naive(40.0), 130, 130.0),
            FakeMeasurement(self._naive(120.0), 118, 118.0),
            FakeMeasurement(self._naive(270.0), 106, 106.0),
            FakeMeasurement(self._naive(285.1), 104, 104.0),
        ]
        candidates = BACKFILL.build_backfill_points_from_graph(
            self.conn,
            graph_raw,
            uom=CONST.GlucoseUnitOfMeasurement.MGDL,
            use_trend_icons=False,
        )

        missing = BACKFILL.select_missing_points(
            candidates, existing, now_ts=self._ts(290.0)
        )
        # +25.1m and +285.1m match live states; +40m, +120m, +270m are selected
        # (even though +40m had unavailable and +120m had a stale repeat!)
        self.assertEqual([p.value_mg_dl for p in missing], [130, 118, 106])

        # Re-running after those points are inserted yields 0 missing points (idempotent)
        after_insert = list(existing) + [
            BACKFILL.ExistingStateRecord(
                state_id=100 + i,
                state=pt.state_str,
                last_updated_ts=pt.timestamp_ts,
                last_changed_ts=None,
                old_state_id=None,
                measurement_ts=pt.timestamp_ts,
            )
            for i, pt in enumerate(missing)
        ]
        second_pass = BACKFILL.select_missing_points(
            candidates, after_insert, now_ts=self._ts(290.0)
        )
        self.assertEqual(second_pass, [])

    def test_identify_bridged_stale_states_vs_real_outage(self) -> None:
        existing = [
            BACKFILL.ExistingStateRecord(1, "135", self._ts(0), None, None, self._ts(0)),
            # Bridged unavailable at +15m (backfill points at +10m and +20m)
            BACKFILL.ExistingStateRecord(2, "unavailable", self._ts(15), None, 1, None),
            # Unbridged unavailable at +180m (no readings between +30m and +300m)
            BACKFILL.ExistingStateRecord(3, "unavailable", self._ts(180), None, 2, None),
            BACKFILL.ExistingStateRecord(4, "100", self._ts(300), None, 3, self._ts(300)),
        ]
        inserted = BACKFILL.build_backfill_points_from_graph(
            self.conn,
            [
                FakeMeasurement(self._naive(10), 132, 132.0),
                FakeMeasurement(self._naive(20), 129, 129.0),
                FakeMeasurement(self._naive(30), 125, 125.0),
            ],
            uom=CONST.GlucoseUnitOfMeasurement.MGDL,
            use_trend_icons=False,
        )
        stale_ids = BACKFILL.identify_bridged_stale_state_ids(existing, inserted)
        self.assertEqual(stale_ids, {2})

    def test_compute_state_chain_updates(self) -> None:
        chain = [
            BACKFILL.ExistingStateRecord(10, "135", 1000.0, None, 9),
            BACKFILL.ExistingStateRecord(50, "130", 1300.0, None, None),
            BACKFILL.ExistingStateRecord(51, "130", 1600.0, None, None),
            BACKFILL.ExistingStateRecord(14, "120", 1900.0, None, 10),
        ]
        updates = BACKFILL.compute_state_chain_updates(chain)
        self.assertEqual(
            updates,
            [
                (50, 10, None),
                (51, 50, 1300.0),
                (14, 51, None),
            ],
        )

    def test_compile_bucket_statistics_5min_and_hourly(self) -> None:
        # 07:00 = 100, 07:15 = 120, 07:30 = 140, 07:45 = 120
        timeline = [
            (self._ts(0), 100.0),
            (self._ts(15), 120.0),
            (self._ts(30), 140.0),
            (self._ts(45), 120.0),
        ]
        short_stats = BACKFILL.compile_bucket_statistics(
            timeline,
            start_dt=self.base_dt,
            end_dt=self.base_dt + timedelta(hours=1),
            bucket_duration=timedelta(minutes=5),
        )
        self.assertEqual(len(short_stats), 12)
        self.assertAlmostEqual(short_stats[0]["mean"], 100.0)
        self.assertAlmostEqual(short_stats[3]["mean"], 120.0)
        self.assertAlmostEqual(short_stats[6]["mean"], 140.0)
        self.assertAlmostEqual(short_stats[9]["mean"], 120.0)

        hourly_stats = BACKFILL.compile_bucket_statistics(
            timeline,
            start_dt=self.base_dt,
            end_dt=self.base_dt + timedelta(hours=1),
            bucket_duration=timedelta(hours=1),
        )
        self.assertEqual(len(hourly_stats), 1)
        # Equal 15-min quarters of 100, 120, 140, 120 => mean 120, min 100, max 140
        self.assertAlmostEqual(hourly_stats[0]["mean"], 120.0)
        self.assertAlmostEqual(hourly_stats[0]["min"], 100.0)
        self.assertAlmostEqual(hourly_stats[0]["max"], 140.0)

    def test_should_trigger_periodic_backfill(self) -> None:
        now = self.base_dt + timedelta(hours=5)
        # Disabled when interval_hours <= 0
        self.assertFalse(BACKFILL.should_trigger_periodic_backfill(None, now, 0))
        self.assertFalse(
            BACKFILL.should_trigger_periodic_backfill(self.base_dt, now, 0)
        )
        # Triggers when never checked and interval_hours > 0
        self.assertTrue(BACKFILL.should_trigger_periodic_backfill(None, now, 4))
        # Does not trigger before interval elapses
        self.assertFalse(
            BACKFILL.should_trigger_periodic_backfill(
                now - timedelta(hours=3, minutes=59), now, 4
            )
        )
        # Triggers once interval elapses
        self.assertTrue(
            BACKFILL.should_trigger_periodic_backfill(
                now - timedelta(hours=4), now, 4
            )
        )

    def test_manifests_translations_and_python_syntax(self) -> None:
        for py_file in COMPONENT_DIR.glob("*.py"):
            ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for json_file in [
            COMPONENT_DIR / "manifest.json",
            COMPONENT_DIR / "strings.json",
            COMPONENT_DIR / "translations" / "en.json",
            COMPONENT_DIR / "translations" / "da.json",
        ]:
            parsed = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertIsInstance(parsed, dict)


if __name__ == "__main__":
    unittest.main()
