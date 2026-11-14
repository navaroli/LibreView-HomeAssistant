"""Historical glucose data backfill into Home Assistant Recorder."""

from __future__ import annotations

import bisect
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .const import (
    DEFAULT_ICON,
    DOMAIN,
    LOGGER,
    TREND_ICONS,
    TREND_MESSAGE,
    GlucoseUnitOfMeasurement,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from LibreView.models import Connection, GlucoseMeasurement


# Proximity window (seconds) within which an existing valid numeric state
# is considered to already represent a graphData reading.
DUPLICATE_PROXIMITY_SECONDS = 90.0

# Readings whose measurement_timestamp lags last_updated_ts by more than this
# threshold were recorded while data was already stale.
STALE_RECORDED_STATE_SECONDS = 900.0

# Maximum gap (seconds) on each side of an unavailable/stale state for
# backfilled points to consider that outage bridged and clean it up.
MAX_BRIDGE_GAP_SECONDS = 1800.0

# Maximum duration (seconds) a single reading carries forward when compiling
# 5-minute / hourly statistics if no newer reading arrives.
MAX_STAT_CARRY_SECONDS = 1800.0


@dataclass(frozen=True, slots=True)
class BackfillPoint:
    """Normalized historical glucose measurement ready for Recorder insertion."""

    timestamp_ts: float
    timestamp_iso: str
    state_str: str
    numeric_value: float
    value_mmol_l: float
    value_mg_dl: int
    target_high_mmol_l: float
    target_low_mmol_l: float
    target_high_mg_dl: int
    target_low_mg_dl: int
    trend: str
    app_uom_str: str
    icon: str

    def to_attributes_dict(
        self,
        *,
        unit_of_measurement: str,
        friendly_name: str,
    ) -> dict[str, Any]:
        """Build the recorded state attributes dictionary matching GlucoseSensor."""
        return {
            "state_class": "measurement",
            "value_mmol_l": self.value_mmol_l,
            "value_mg_dl": self.value_mg_dl,
            "target_high_mmol_l": self.target_high_mmol_l,
            "target_low_mmol_l": self.target_low_mmol_l,
            "target_high_mg_dl": self.target_high_mg_dl,
            "target_low_mg_dl": self.target_low_mg_dl,
            "trend": self.trend,
            "app_unit_of_measurement": self.app_uom_str,
            "measurement_timestamp": self.timestamp_iso,
            "unit_of_measurement": unit_of_measurement,
            "icon": self.icon,
            "friendly_name": friendly_name,
        }


@dataclass(slots=True)
class ExistingStateRecord:
    """Lightweight representation of a Recorder states row."""

    state_id: int
    state: str | None
    last_updated_ts: float
    last_changed_ts: float | None
    old_state_id: int | None
    measurement_ts: float | None = None


def parse_measurement_ts_from_attrs(shared_attrs: str | None) -> float | None:
    """Extract measurement_timestamp (as UTC epoch seconds) from shared_attrs JSON."""
    if not shared_attrs:
        return None
    try:
        data = json.loads(shared_attrs)
        raw_ts = data.get("measurement_timestamp")
        if not raw_ts or not isinstance(raw_ts, str):
            return None
        dt = datetime.fromisoformat(raw_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def build_backfill_points_from_graph(
    connection: Connection,
    graph_measurements: Sequence[GlucoseMeasurement],
    *,
    uom: GlucoseUnitOfMeasurement,
    use_trend_icons: bool,
) -> list[BackfillPoint]:
    """Convert LibreView graphData measurements into sorted BackfillPoints."""
    points_by_ts: dict[float, BackfillPoint] = {}
    app_uom = connection.uom
    app_uom_str = "mmol/L" if app_uom == 0 else "mg/dL"
    target_high_mg_dl = connection.target_high
    target_low_mg_dl = connection.target_low
    target_high_mmol_l = round(target_high_mg_dl / 18, 1)
    target_low_mmol_l = round(target_low_mg_dl / 18, 1)

    for m in graph_measurements:
        try:
            dt_utc = m.factory_timestamp.replace(tzinfo=timezone.utc)
        except (AttributeError, TypeError, ValueError):
            continue

        ts = dt_utc.timestamp()
        value_mg_dl = int(m.value_in_mg_per_dl)
        value_mmol_l = (value_mg_dl / 18.0) if app_uom == 1 else float(m.value)

        if uom == GlucoseUnitOfMeasurement.MMOLL:
            native_val: int | float = value_mmol_l
        else:
            native_val = value_mg_dl

        numeric_val = float(native_val)
        if not math.isfinite(numeric_val):
            continue

        trend_arrow = getattr(m, "trend_arrow", None)
        trend = TREND_MESSAGE.get(trend_arrow, "unknown")
        icon = (
            TREND_ICONS.get(trend_arrow, DEFAULT_ICON)
            if use_trend_icons
            else DEFAULT_ICON
        )

        points_by_ts[ts] = BackfillPoint(
            timestamp_ts=ts,
            timestamp_iso=dt_utc.isoformat(),
            state_str=str(native_val),
            numeric_value=numeric_val,
            value_mmol_l=value_mmol_l,
            value_mg_dl=value_mg_dl,
            target_high_mmol_l=target_high_mmol_l,
            target_low_mmol_l=target_low_mmol_l,
            target_high_mg_dl=target_high_mg_dl,
            target_low_mg_dl=target_low_mg_dl,
            trend=trend,
            app_uom_str=app_uom_str,
            icon=icon,
        )

    return [points_by_ts[k] for k in sorted(points_by_ts)]


def is_genuine_numeric_state(
    rec: ExistingStateRecord,
    *,
    max_stale_seconds: float = STALE_RECORDED_STATE_SECONDS,
) -> tuple[bool, float | None]:
    """Return (True, float_value) if rec is a non-stale numeric sensor state."""
    if rec.state is None:
        return False, None
    try:
        val = float(rec.state)
    except (ValueError, TypeError):
        return False, None
    if not math.isfinite(val):
        return False, None
    if (
        rec.measurement_ts is not None
        and (rec.last_updated_ts - rec.measurement_ts) > max_stale_seconds
    ):
        return False, None
    return True, val


def _has_nearby_timestamp(
    sorted_timestamps: Sequence[float],
    target_ts: float,
    tolerance: float,
) -> bool:
    """Return True if any entry in sorted_timestamps is within tolerance of target_ts."""
    if not sorted_timestamps:
        return False
    idx = bisect.bisect_left(sorted_timestamps, target_ts)
    if idx < len(sorted_timestamps) and abs(sorted_timestamps[idx] - target_ts) <= tolerance:
        return True
    if idx > 0 and abs(sorted_timestamps[idx - 1] - target_ts) <= tolerance:
        return True
    return False


def select_missing_points(
    points: Sequence[BackfillPoint],
    existing_records: Sequence[ExistingStateRecord],
    *,
    now_ts: float,
    proximity_seconds: float = DUPLICATE_PROXIMITY_SECONDS,
) -> list[BackfillPoint]:
    """Filter candidate BackfillPoints down to readings missing from Recorder."""
    ref_timestamps_set: set[float] = set()
    for rec in existing_records:
        is_valid, _ = is_genuine_numeric_state(rec)
        if is_valid:
            ref_timestamps_set.add(rec.last_updated_ts)
            if rec.measurement_ts is not None:
                ref_timestamps_set.add(rec.measurement_ts)

    sorted_refs = sorted(ref_timestamps_set)
    selected: list[BackfillPoint] = []
    for pt in sorted(points, key=lambda p: p.timestamp_ts):
        if pt.timestamp_ts > now_ts + 60.0:
            continue
        if _has_nearby_timestamp(sorted_refs, pt.timestamp_ts, proximity_seconds):
            continue
        if selected and abs(pt.timestamp_ts - selected[-1].timestamp_ts) < 30.0:
            continue
        selected.append(pt)

    return selected


def identify_bridged_stale_state_ids(
    existing_records: Sequence[ExistingStateRecord],
    inserted_points: Sequence[BackfillPoint],
    *,
    max_bridge_gap_seconds: float = MAX_BRIDGE_GAP_SECONDS,
) -> set[int]:
    """Identify unavailable/stale state rows bridged on both sides by valid data."""
    if not inserted_points:
        return set()

    valid_ts_set: set[float] = {pt.timestamp_ts for pt in inserted_points}
    for rec in existing_records:
        is_valid, _ = is_genuine_numeric_state(rec)
        if is_valid:
            valid_ts_set.add(rec.last_updated_ts)

    valid_ts = sorted(valid_ts_set)
    if len(valid_ts) < 2:
        return set()

    min_window = inserted_points[0].timestamp_ts - max_bridge_gap_seconds
    max_window = inserted_points[-1].timestamp_ts + max_bridge_gap_seconds

    to_delete: set[int] = set()
    for rec in existing_records:
        if rec.last_updated_ts < min_window or rec.last_updated_ts > max_window:
            continue
        is_valid, _ = is_genuine_numeric_state(rec)
        if is_valid:
            continue

        idx = bisect.bisect_right(valid_ts, rec.last_updated_ts)
        prev_ok = (
            idx > 0
            and (rec.last_updated_ts - valid_ts[idx - 1]) <= max_bridge_gap_seconds
        )
        next_ok = (
            idx < len(valid_ts)
            and (valid_ts[idx] - rec.last_updated_ts) <= max_bridge_gap_seconds
        )
        if prev_ok and next_ok:
            to_delete.add(rec.state_id)

    return to_delete


def compute_state_chain_updates(
    ordered_records: Sequence[ExistingStateRecord],
) -> list[tuple[int, int | None, float | None]]:
    """Compute (state_id, old_state_id, last_changed_ts) repairs for ordered rows."""
    if len(ordered_records) < 2:
        return []

    updates: list[tuple[int, int | None, float | None]] = []
    first = ordered_records[0]
    run_start_ts = (
        first.last_changed_ts
        if first.last_changed_ts is not None
        else first.last_updated_ts
    )

    for idx in range(1, len(ordered_records)):
        prev = ordered_records[idx - 1]
        cur = ordered_records[idx]
        expected_old_id = prev.state_id
        if cur.state != prev.state:
            expected_last_changed_ts = None
            run_start_ts = cur.last_updated_ts
        else:
            expected_last_changed_ts = run_start_ts

        last_changed_differs = (
            (cur.last_changed_ts is None) != (expected_last_changed_ts is None)
            or (
                cur.last_changed_ts is not None
                and expected_last_changed_ts is not None
                and abs(cur.last_changed_ts - expected_last_changed_ts) > 1e-6
            )
        )
        if cur.old_state_id != expected_old_id or last_changed_differs:
            updates.append((cur.state_id, expected_old_id, expected_last_changed_ts))

    return updates


def compile_bucket_statistics(
    timeline: Sequence[tuple[float, float | None]],
    *,
    start_dt: datetime,
    end_dt: datetime,
    bucket_duration: timedelta,
    max_carry_seconds: float = MAX_STAT_CARRY_SECONDS,
) -> list[dict[str, Any]]:
    """Compile time-weighted mean/min/max statistics across [start_dt, end_dt).

    Each entry in timeline is (timestamp_ts, numeric_value_or_None) sorted by
    timestamp_ts ascending. None represents an unavailable/unknown state.
    """
    if start_dt >= end_dt or not timeline:
        return []

    # Expand timeline with synthetic None cutoffs when gaps exceed max_carry_seconds
    expanded: list[tuple[float, float | None]] = []
    for i, (ts, val) in enumerate(timeline):
        expanded.append((ts, val))
        if val is not None and max_carry_seconds > 0:
            expire_ts = ts + max_carry_seconds
            next_ts = timeline[i + 1][0] if (i + 1) < len(timeline) else float("inf")
            if expire_ts < next_ts:
                expanded.append((expire_ts, None))

    ts_list = [item[0] for item in expanded]
    bucket_sec = bucket_duration.total_seconds()
    results: list[dict[str, Any]] = []

    cur_dt = start_dt
    while cur_dt + bucket_duration <= end_dt:
        b_start = cur_dt.timestamp()
        b_end = b_start + bucket_sec

        # Find state active at b_start (latest entry with ts <= b_start)
        start_idx = bisect.bisect_right(ts_list, b_start) - 1
        end_idx = bisect.bisect_left(ts_list, b_end)

        weighted_sum = 0.0
        total_weight = 0.0
        vals: list[float] = []

        pos_ts = b_start
        active_val = expanded[start_idx][1] if start_idx >= 0 else None

        for idx in range(max(0, start_idx + 1), end_idx):
            ev_ts, ev_val = expanded[idx]
            if active_val is not None and ev_ts > pos_ts:
                dur = ev_ts - pos_ts
                weighted_sum += active_val * dur
                total_weight += dur
                vals.append(active_val)
            pos_ts = ev_ts
            active_val = ev_val
            if active_val is not None:
                vals.append(active_val)

        if active_val is not None and b_end > pos_ts:
            dur = b_end - pos_ts
            weighted_sum += active_val * dur
            total_weight += dur
            vals.append(active_val)

        if vals:
            mean_val = (
                (weighted_sum / total_weight)
                if total_weight > 0
                else (sum(vals) / len(vals))
            )
            results.append(
                {
                    "start": cur_dt,
                    "mean": mean_val,
                    "min": min(vals),
                    "max": max(vals),
                }
            )

        cur_dt += bucket_duration

    return results


def _floor_dt(dt: datetime, minutes: int) -> datetime:
    """Floor a UTC datetime to an exact minute boundary (e.g., 5 or 60)."""
    if minutes == 60:
        return dt.replace(minute=0, second=0, microsecond=0)
    floored_min = (dt.minute // minutes) * minutes
    return dt.replace(minute=floored_min, second=0, microsecond=0)


def _ceil_dt(dt: datetime, minutes: int) -> datetime:
    """Ceil a UTC datetime to an exact minute boundary."""
    floored = _floor_dt(dt, minutes)
    if floored == dt:
        return dt
    return floored + timedelta(minutes=minutes)


def should_trigger_periodic_backfill(
    last_check: datetime | None,
    now: datetime,
    interval_hours: int,
) -> bool:
    """Return True if the periodic safety-net backfill interval has elapsed."""
    if interval_hours <= 0:
        return False
    if last_check is None:
        return True
    return (now - last_check) >= timedelta(hours=interval_hours)


def _execute_recorder_backfill(
    instance: Any,
    *,
    entity_id: str,
    points: list[BackfillPoint],
    unit_of_measurement: str,
    friendly_name: str,
) -> int:
    """Run on the Recorder thread to backfill States, ShortTerm, and Hourly stats."""
    from homeassistant.components.recorder import DOMAIN as RECORDER_DOMAIN
    from homeassistant.components.recorder.db_schema import (
        StateAttributes,
        States,
        StatesMeta,
        Statistics,
        StatisticsShortTerm,
    )
    from homeassistant.components.recorder.models import (
        StatisticMeanType,
        StatisticMetaData,
    )
    from homeassistant.components.recorder.statistics import import_statistics
    from homeassistant.components.recorder.util import session_scope
    from homeassistant.util.json import json_bytes

    if not points:
        return 0

    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()
    first_candidate_ts = points[0].timestamp_ts

    inserted_count = 0
    latest_state_id: int | None = None
    short_term_stats: list[Any] = []
    hourly_stats: list[Any] = []
    stat_metadata: StatisticMetaData | None = None

    with session_scope(session=instance.get_session()) as session:
        metadata_id = instance.states_meta_manager.get(entity_id, session, True)
        if metadata_id is None:
            states_meta = StatesMeta(entity_id=entity_id)
            session.add(states_meta)
            session.flush()
            metadata_id = states_meta.metadata_id
            instance.states_meta_manager._id_map[entity_id] = metadata_id

        # Fetch anchor row strictly before first_candidate_ts plus all rows from first_candidate_ts onward
        anchor_row = (
            session.query(
                States.state_id,
                States.state,
                States.last_updated_ts,
                States.last_changed_ts,
                States.old_state_id,
                StateAttributes.shared_attrs,
            )
            .outerjoin(
                StateAttributes, States.attributes_id == StateAttributes.attributes_id
            )
            .filter(
                States.metadata_id == metadata_id,
                States.last_updated_ts < (first_candidate_ts - DUPLICATE_PROXIMITY_SECONDS),
            )
            .order_by(States.last_updated_ts.desc(), States.state_id.desc())
            .first()
        )

        window_rows = (
            session.query(
                States.state_id,
                States.state,
                States.last_updated_ts,
                States.last_changed_ts,
                States.old_state_id,
                StateAttributes.shared_attrs,
            )
            .outerjoin(
                StateAttributes, States.attributes_id == StateAttributes.attributes_id
            )
            .filter(
                States.metadata_id == metadata_id,
                States.last_updated_ts >= (first_candidate_ts - DUPLICATE_PROXIMITY_SECONDS),
            )
            .order_by(States.last_updated_ts.asc(), States.state_id.asc())
            .all()
        )

        existing_records: list[ExistingStateRecord] = []
        for r in ([anchor_row] if anchor_row is not None else []) + list(window_rows):
            if r.last_updated_ts is None:
                continue
            existing_records.append(
                ExistingStateRecord(
                    state_id=r.state_id,
                    state=r.state,
                    last_updated_ts=float(r.last_updated_ts),
                    last_changed_ts=(
                        float(r.last_changed_ts)
                        if r.last_changed_ts is not None
                        else None
                    ),
                    old_state_id=r.old_state_id,
                    measurement_ts=parse_measurement_ts_from_attrs(r.shared_attrs),
                )
            )

        missing_points = select_missing_points(
            points, existing_records, now_ts=now_ts
        )
        if not missing_points:
            return 0

        stale_ids = identify_bridged_stale_state_ids(existing_records, missing_points)
        if stale_ids:
            session.query(States).filter(States.old_state_id.in_(stale_ids)).update(
                {States.old_state_id: None}, synchronize_session=False
            )
            session.flush()
            session.query(States).filter(States.state_id.in_(stale_ids)).delete(
                synchronize_session=False
            )
            session.flush()
            instance.states_manager.evict_purged_state_ids(stale_ids)

        # Insert StateAttributes and States for each missing point
        attrs_cache: dict[str, int] = {}
        new_states: list[States] = []
        for pt in missing_points:
            attrs_dict = pt.to_attributes_dict(
                unit_of_measurement=unit_of_measurement,
                friendly_name=friendly_name,
            )
            attrs_bytes = json_bytes(attrs_dict)
            shared_attrs = attrs_bytes.decode("utf-8")
            attributes_id = attrs_cache.get(shared_attrs)
            if attributes_id is None:
                attributes_id = instance.state_attributes_manager.get_from_cache(
                    shared_attrs
                )
            if attributes_id is None:
                hash_ = StateAttributes.hash_shared_attrs_bytes(attrs_bytes)
                found = (
                    session.query(StateAttributes.attributes_id)
                    .filter(
                        StateAttributes.hash == hash_,
                        StateAttributes.shared_attrs == shared_attrs,
                    )
                    .first()
                )
                if found is not None:
                    attributes_id = int(found.attributes_id)
                else:
                    db_attrs = StateAttributes(shared_attrs=shared_attrs, hash=hash_)
                    session.add(db_attrs)
                    session.flush()
                    attributes_id = int(db_attrs.attributes_id)
            attrs_cache[shared_attrs] = attributes_id

            new_states.append(
                States(
                    metadata_id=metadata_id,
                    state=pt.state_str,
                    attributes_id=attributes_id,
                    last_updated_ts=pt.timestamp_ts,
                    last_changed_ts=None,
                    last_reported_ts=pt.timestamp_ts,
                    origin_idx=0,
                )
            )

        session.add_all(new_states)
        session.flush()
        inserted_count = len(new_states)

        min_touched_ts = missing_points[0].timestamp_ts - MAX_BRIDGE_GAP_SECONDS
        max_touched_ts = missing_points[-1].timestamp_ts + MAX_BRIDGE_GAP_SECONDS

        # Reload ordered rows from the pre-window anchor through the end of the table
        chain_anchor = (
            session.query(
                States.state_id,
                States.state,
                States.last_updated_ts,
                States.last_changed_ts,
                States.old_state_id,
            )
            .filter(
                States.metadata_id == metadata_id,
                States.last_updated_ts < min_touched_ts,
            )
            .order_by(States.last_updated_ts.desc(), States.state_id.desc())
            .first()
        )
        chain_tail = (
            session.query(
                States.state_id,
                States.state,
                States.last_updated_ts,
                States.last_changed_ts,
                States.old_state_id,
            )
            .filter(
                States.metadata_id == metadata_id,
                States.last_updated_ts >= min_touched_ts,
            )
            .order_by(States.last_updated_ts.asc(), States.state_id.asc())
            .all()
        )

        ordered_chain: list[ExistingStateRecord] = []
        for r in ([chain_anchor] if chain_anchor is not None else []) + list(chain_tail):
            if r.last_updated_ts is None:
                continue
            ordered_chain.append(
                ExistingStateRecord(
                    state_id=r.state_id,
                    state=r.state,
                    last_updated_ts=float(r.last_updated_ts),
                    last_changed_ts=(
                        float(r.last_changed_ts)
                        if r.last_changed_ts is not None
                        else None
                    ),
                    old_state_id=r.old_state_id,
                )
            )

        chain_updates = compute_state_chain_updates(ordered_chain)
        for state_id, expected_old_id, expected_last_changed_ts in chain_updates:
            session.query(States).filter(States.state_id == state_id).update(
                {
                    States.old_state_id: expected_old_id,
                    States.last_changed_ts: expected_last_changed_ts,
                },
                synchronize_session=False,
            )
        if ordered_chain:
            latest_state_id = ordered_chain[-1].state_id

        session.flush()

        # Prepare 5-minute and hourly statistics across affected completed periods
        stats_start_dt = _floor_dt(
            datetime.fromtimestamp(missing_points[0].timestamp_ts, tz=timezone.utc),
            60,
        )
        max_point_dt = datetime.fromtimestamp(
            max(missing_points[-1].timestamp_ts, max_touched_ts), tz=timezone.utc
        )
        short_end_dt = min(_ceil_dt(max_point_dt, 5), _floor_dt(now_utc, 5))
        hour_end_dt = min(_ceil_dt(max_point_dt, 60), _floor_dt(now_utc, 60))
        query_end_dt = max(short_end_dt, hour_end_dt)

        if query_end_dt > stats_start_dt:
            stat_anchor = (
                session.query(
                    States.state_id,
                    States.state,
                    States.last_updated_ts,
                    StateAttributes.shared_attrs,
                )
                .outerjoin(
                    StateAttributes,
                    States.attributes_id == StateAttributes.attributes_id,
                )
                .filter(
                    States.metadata_id == metadata_id,
                    States.last_updated_ts <= stats_start_dt.timestamp(),
                )
                .order_by(States.last_updated_ts.desc(), States.state_id.desc())
                .first()
            )
            stat_rows = (
                session.query(
                    States.state_id,
                    States.state,
                    States.last_updated_ts,
                    StateAttributes.shared_attrs,
                )
                .outerjoin(
                    StateAttributes,
                    States.attributes_id == StateAttributes.attributes_id,
                )
                .filter(
                    States.metadata_id == metadata_id,
                    States.last_updated_ts > stats_start_dt.timestamp(),
                    States.last_updated_ts < query_end_dt.timestamp(),
                )
                .order_by(States.last_updated_ts.asc(), States.state_id.asc())
                .all()
            )

            timeline: list[tuple[float, float | None]] = []
            for r in ([stat_anchor] if stat_anchor is not None else []) + list(stat_rows):
                if r.last_updated_ts is None:
                    continue
                rec = ExistingStateRecord(
                    state_id=r.state_id,
                    state=r.state,
                    last_updated_ts=float(r.last_updated_ts),
                    last_changed_ts=None,
                    old_state_id=None,
                    measurement_ts=parse_measurement_ts_from_attrs(r.shared_attrs),
                )
                is_valid, val = is_genuine_numeric_state(rec)
                timeline.append((rec.last_updated_ts, val if is_valid else None))

            if short_end_dt > stats_start_dt:
                short_term_stats = compile_bucket_statistics(
                    timeline,
                    start_dt=stats_start_dt,
                    end_dt=short_end_dt,
                    bucket_duration=timedelta(minutes=5),
                )
            if hour_end_dt > stats_start_dt:
                hourly_stats = compile_bucket_statistics(
                    timeline,
                    start_dt=stats_start_dt,
                    end_dt=hour_end_dt,
                    bucket_duration=timedelta(hours=1),
                )

            old_meta = instance.statistics_meta_manager.get_many(
                session, statistic_ids={entity_id}
            )
            if entity_id in old_meta:
                existing_meta = old_meta[entity_id][1]
                stat_metadata = {
                    "mean_type": StatisticMeanType.ARITHMETIC,
                    "has_sum": bool(existing_meta.get("has_sum", False)),
                    "name": existing_meta.get("name"),
                    "source": RECORDER_DOMAIN,
                    "statistic_id": entity_id,
                    "unit_class": existing_meta.get("unit_class"),
                    "unit_of_measurement": existing_meta.get(
                        "unit_of_measurement", unit_of_measurement
                    ),
                }
                stat_meta_id = old_meta[entity_id][0]
                # Remove any stale statistics buckets inside our recompiled window
                # that no longer have valid active measurements
                if short_end_dt > stats_start_dt:
                    valid_short_ts = {s["start"].timestamp() for s in short_term_stats}
                    existing_short = (
                        session.query(
                            StatisticsShortTerm.id, StatisticsShortTerm.start_ts
                        )
                        .filter(
                            StatisticsShortTerm.metadata_id == stat_meta_id,
                            StatisticsShortTerm.start_ts >= stats_start_dt.timestamp(),
                            StatisticsShortTerm.start_ts < short_end_dt.timestamp(),
                        )
                        .all()
                    )
                    obsolete_short_ids = [
                        row.id
                        for row in existing_short
                        if row.start_ts is not None
                        and not _has_nearby_timestamp(
                            sorted(valid_short_ts), float(row.start_ts), 0.5
                        )
                    ]
                    if obsolete_short_ids:
                        session.query(StatisticsShortTerm).filter(
                            StatisticsShortTerm.id.in_(obsolete_short_ids)
                        ).delete(synchronize_session=False)

                if hour_end_dt > stats_start_dt:
                    valid_hour_ts = {s["start"].timestamp() for s in hourly_stats}
                    existing_hour = (
                        session.query(Statistics.id, Statistics.start_ts)
                        .filter(
                            Statistics.metadata_id == stat_meta_id,
                            Statistics.start_ts >= stats_start_dt.timestamp(),
                            Statistics.start_ts < hour_end_dt.timestamp(),
                        )
                        .all()
                    )
                    obsolete_hour_ids = [
                        row.id
                        for row in existing_hour
                        if row.start_ts is not None
                        and not _has_nearby_timestamp(
                            sorted(valid_hour_ts), float(row.start_ts), 0.5
                        )
                    ]
                    if obsolete_hour_ids:
                        session.query(Statistics).filter(
                            Statistics.id.in_(obsolete_hour_ids)
                        ).delete(synchronize_session=False)
            else:
                stat_metadata = {
                    "mean_type": StatisticMeanType.ARITHMETIC,
                    "has_sum": False,
                    "name": None,
                    "source": RECORDER_DOMAIN,
                    "statistic_id": entity_id,
                    "unit_class": None,
                    "unit_of_measurement": unit_of_measurement,
                }

    if latest_state_id is not None:
        instance.states_manager.pop_pending(entity_id)
        instance.states_manager._last_committed_id[entity_id] = latest_state_id

    if stat_metadata is not None:
        if short_term_stats:
            import_statistics(
                instance, stat_metadata, short_term_stats, StatisticsShortTerm
            )
        if hourly_stats:
            import_statistics(instance, stat_metadata, hourly_stats, Statistics)

    LOGGER.info(
        "Backfilled %d historical glucose reading(s) for %s",
        inserted_count,
        entity_id,
    )
    return inserted_count


async def async_queue_recorder_backfill(
    hass: HomeAssistant,
    *,
    connection_id: UUID,
    connection: Connection,
    graph_measurements: Sequence[GlucoseMeasurement],
    uom: GlucoseUnitOfMeasurement,
    use_trend_icons: bool,
) -> bool:
    """Resolve the glucose sensor entity and queue a backfill task in Recorder."""
    if not graph_measurements:
        return False

    try:
        from homeassistant.components.recorder import get_instance, is_entity_recorded
        from homeassistant.components.recorder.tasks import RecorderTask
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers.recorder import DATA_INSTANCE
    except ImportError:
        LOGGER.debug("Recorder component unavailable; skipping backfill")
        return False

    if DATA_INSTANCE not in hass.data:
        LOGGER.debug("Recorder is not running; skipping backfill")
        return False

    ent_reg = er.async_get(hass)
    unique_id = f"{connection_id}_glucose_reading"
    entity_id = ent_reg.async_get_entity_id("sensor", DOMAIN, unique_id)
    if not entity_id:
        LOGGER.debug(
            "Entity for unique_id %s not registered yet; deferring backfill",
            unique_id,
        )
        return False

    if not is_entity_recorded(hass, entity_id):
        LOGGER.debug("Entity %s is excluded from Recorder; skipping backfill", entity_id)
        return False

    points = build_backfill_points_from_graph(
        connection,
        graph_measurements,
        uom=uom,
        use_trend_icons=use_trend_icons,
    )
    if not points:
        return False

    default_name = f"{connection.first_name} {connection.last_name} glucose level"
    live_state = hass.states.get(entity_id)
    friendly_name = (
        live_state.attributes.get("friendly_name", default_name)
        if live_state is not None
        else default_name
    )
    unit_str = (
        live_state.attributes.get("unit_of_measurement", uom.value)
        if live_state is not None
        else uom.value
    )

    @dataclass(slots=True)
    class LibreViewBackfillTask(RecorderTask):
        """Recorder queue task to backfill historical LibreView measurements."""

        target_entity_id: str
        backfill_points: list[BackfillPoint]
        uom_str: str
        entity_friendly_name: str

        def run(self, instance: Any) -> None:
            try:
                _execute_recorder_backfill(
                    instance,
                    entity_id=self.target_entity_id,
                    points=self.backfill_points,
                    unit_of_measurement=self.uom_str,
                    friendly_name=self.entity_friendly_name,
                )
            except Exception:
                logging.getLogger(__package__).exception(
                    "Failed backfilling historical glucose data for %s",
                    self.target_entity_id,
                )

    recorder = get_instance(hass)
    recorder.queue_task(
        LibreViewBackfillTask(
            target_entity_id=entity_id,
            backfill_points=points,
            uom_str=unit_str,
            entity_friendly_name=friendly_name,
        )
    )
    return True
