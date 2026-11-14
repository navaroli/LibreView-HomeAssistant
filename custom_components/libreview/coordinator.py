import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_EMAIL, CONF_PASSWORD
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from LibreView import LibreView
from LibreView.models import Connection

from .backfill import async_queue_recorder_backfill, should_trigger_periodic_backfill
from .const import (
    BACKFILL_GAP_THRESHOLD,
    BACKFILL_RETRY_OFFSETS,
    CONF_BACKFILL,
    CONF_BACKFILL_INTERVAL_HOURS,
    CONF_SHOW_TREND_ARROW,
    CONF_UOM,
    DEFAULT_BACKFILL,
    DEFAULT_BACKFILL_INTERVAL_HOURS,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    LOGGER,
    STARTUP_BACKFILL_OFFSETS,
    GlucoseUnitOfMeasurement,
)


class LibreViewCoordinator(DataUpdateCoordinator):
    entry: ConfigEntry
    libre: LibreView

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.entry = entry
        region = entry.data.get("region", None)
        region = region if region and len(region) == 2 else None
        self.libre = LibreView(
            username=entry.data[CONF_EMAIL],
            password=entry.data[CONF_PASSWORD],
            region=region,
        )
        self.backfill_enabled = bool(
            entry.data.get(CONF_BACKFILL, DEFAULT_BACKFILL)
        )
        self.backfill_interval_hours = max(
            0,
            int(
                entry.data.get(
                    CONF_BACKFILL_INTERVAL_HOURS, DEFAULT_BACKFILL_INTERVAL_HOURS
                )
            ),
        )
        self.uom = GlucoseUnitOfMeasurement.from_str(entry.data[CONF_UOM])
        self.show_trend_arrow = bool(entry.data.get(CONF_SHOW_TREND_ARROW, False))
        self._entities_ready = False
        self._last_measurement_ts: dict[UUID, datetime] = {}
        self._last_backfill_check: dict[UUID, datetime] = {}
        self._pending_backfill_countdowns: dict[UUID, set[int]] = {}
        self._backfill_locks: dict[UUID, asyncio.Lock] = {}

        super().__init__(
            hass, LOGGER, name=DOMAIN, update_interval=DEFAULT_SCAN_INTERVAL
        )

    @staticmethod
    def _extract_measurement_ts(connection: Connection) -> datetime | None:
        try:
            return connection.glucose_measurement.factory_timestamp.replace(
                tzinfo=timezone.utc
            )
        except (AttributeError, TypeError, ValueError):
            return None

    @callback
    def async_schedule_startup_backfill(self) -> None:
        """Mark platform entities ready and trigger an initial 12-hour backfill check."""
        self._entities_ready = True
        if not self.backfill_enabled:
            return

        now_utc = datetime.now(timezone.utc)
        for connection_id, connection in self.libre.connections_dict.items():
            self._last_backfill_check[connection_id] = now_utc
            offsets = set(STARTUP_BACKFILL_OFFSETS)
            if 0 in offsets:
                offsets.discard(0)
                self.hass.async_create_task(
                    self._async_backfill_connection(connection_id, connection)
                )
            if offsets:
                existing = self._pending_backfill_countdowns.setdefault(
                    connection_id, set()
                )
                existing.update(offsets)

    async def _async_backfill_connection(
        self, connection_id: UUID, connection: Connection
    ) -> None:
        """Fetch historical graphData for a connection and queue Recorder backfill."""
        lock = self._backfill_locks.setdefault(connection_id, asyncio.Lock())
        if lock.locked():
            return

        async with lock:
            self._last_backfill_check[connection_id] = datetime.now(timezone.utc)
            try:
                graph_data = await self.hass.async_add_executor_job(
                    self.libre.get_graph_from_connection, connection
                )
            except Exception as ex:  # pylint: disable=broad-except
                LOGGER.warning(
                    "Could not fetch historical graph data for %s %s: %s",
                    connection.first_name,
                    connection.last_name,
                    ex,
                )
                return

            await async_queue_recorder_backfill(
                self.hass,
                connection_id=connection_id,
                connection=connection,
                graph_measurements=graph_data,
                uom=self.uom,
                use_trend_icons=self.show_trend_arrow,
            )

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            LOGGER.debug("Updating data")
            await self.hass.async_add_executor_job(self.libre.get_connections)
        except Exception as ex:
            LOGGER.error(ex)
            raise

        if self.backfill_enabled:
            now_utc = datetime.now(timezone.utc)
            for connection_id, connection in self.libre.connections_dict.items():
                current_ts = self._extract_measurement_ts(connection)
                prev_ts = self._last_measurement_ts.get(connection_id)

                existing_countdowns = self._pending_backfill_countdowns.get(
                    connection_id, set()
                )
                next_countdowns = {c - 1 for c in existing_countdowns if c > 0}

                if (
                    current_ts is not None
                    and prev_ts is not None
                    and (current_ts - prev_ts) >= BACKFILL_GAP_THRESHOLD
                ):
                    gap_mins = (current_ts - prev_ts).total_seconds() / 60.0
                    LOGGER.info(
                        "Detected %.1f-minute glucose data gap for %s %s; scheduling historical backfill",
                        gap_mins,
                        connection.first_name,
                        connection.last_name,
                    )
                    next_countdowns.update(BACKFILL_RETRY_OFFSETS)

                if self._entities_ready and should_trigger_periodic_backfill(
                    self._last_backfill_check.get(connection_id),
                    now_utc,
                    self.backfill_interval_hours,
                ):
                    LOGGER.debug(
                        "Triggering periodic %dh safety-net backfill check for %s %s",
                        self.backfill_interval_hours,
                        connection.first_name,
                        connection.last_name,
                    )
                    next_countdowns.add(0)

                if current_ts is not None:
                    self._last_measurement_ts[connection_id] = current_ts

                if self._entities_ready and 0 in next_countdowns:
                    next_countdowns.discard(0)
                    self._last_backfill_check[connection_id] = now_utc
                    self.hass.async_create_task(
                        self._async_backfill_connection(connection_id, connection)
                    )

                if next_countdowns:
                    self._pending_backfill_countdowns[connection_id] = next_countdowns
                else:
                    self._pending_backfill_countdowns.pop(connection_id, None)

        # Store data in a way Home Assistant can easily consume it
        LOGGER.debug(self.libre.connections_dict)
        data: dict[str, Any] = {
            "glucose_readings": self.libre.connections_dict,
        }
        return data
