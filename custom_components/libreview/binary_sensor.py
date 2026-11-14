from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from uuid import UUID

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from LibreView.models import Connection, GlucoseMeasurement

from .const import (
    CONF_MAX_DATA_AGE,
    CONF_UOM,
    DEFAULT_ICON,
    DEFAULT_MAX_DATA_AGE,
    DOMAIN,
    LOGGER,
    GlucoseUnitOfMeasurement,
)
from .coordinator import LibreViewCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: LibreViewCoordinator = hass.data[DOMAIN][entry.entry_id]
    uom = GlucoseUnitOfMeasurement.from_str(entry.data[CONF_UOM])
    max_data_age = int(entry.data.get(CONF_MAX_DATA_AGE, DEFAULT_MAX_DATA_AGE))

    sensors = [
        GlucoseHighSensor(coordinator, connection_id, uom, max_data_age)
        for connection_id, _ in coordinator.data["glucose_readings"].items()
    ] + [
        GlucoseLowSensor(coordinator, connection_id, uom, max_data_age)
        for connection_id, _ in coordinator.data["glucose_readings"].items()
    ]
    async_add_entities(sensors)


class _GlucoseThresholdBinarySensor(CoordinatorEntity, BinarySensorEntity):
    coordinator: LibreViewCoordinator
    uom: GlucoseUnitOfMeasurement

    def __init__(
        self,
        coordinator: LibreViewCoordinator,
        connection_id: UUID,
        uom: GlucoseUnitOfMeasurement,
        max_data_age: int = DEFAULT_MAX_DATA_AGE,
    ):
        super().__init__(coordinator)
        self.connection_id = connection_id
        self.uom = uom
        self.max_data_age = max_data_age

    @property
    def icon(self):
        return DEFAULT_ICON

    @property
    def available(self) -> bool:
        if not super().available:
            return False
        if not self.max_data_age:
            return True
        measurement_dt = self._measurement_timestamp
        if measurement_dt is None:
            return True
        return (datetime.now(timezone.utc) - measurement_dt) <= timedelta(
            minutes=self.max_data_age
        )

    @property
    def _measurement_timestamp(self) -> Optional[datetime]:
        try:
            return self.gcm.factory_timestamp.replace(tzinfo=timezone.utc)
        except (AttributeError, TypeError, ValueError, IndexError) as ex:
            LOGGER.debug("Could not determine glucose measurement age: %s", ex)
            return None

    @property
    def connection(self) -> Connection:
        return self.coordinator.data["glucose_readings"][self.connection_id]

    @property
    def gcm(self) -> GlucoseMeasurement:
        return self.connection.glucose_measurement

    @property
    def trend_arrow(self) -> int:
        return self.gcm.trend_arrow


class GlucoseHighSensor(_GlucoseThresholdBinarySensor):
    def __init__(
        self,
        coordinator: LibreViewCoordinator,
        connection_id: UUID,
        uom: GlucoseUnitOfMeasurement,
        max_data_age: int = DEFAULT_MAX_DATA_AGE,
    ):
        super().__init__(coordinator, connection_id, uom, max_data_age)
        self._attr_unique_id = f"{connection_id}_glucose_reading_is_high"

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        name = f"{self.connection.first_name } {self.connection.last_name}"
        return f"{name} glucose level is high"

    @property
    def is_on(self) -> bool:
        """Return the state of the entity."""
        return self.gcm.value_in_mg_per_dl > self.connection.target_high

    @property
    def extra_state_attributes(self) -> Dict[str, int | float]:
        return {
            "target_high_mmol_l": round(self.connection.target_high / 18, 1),
            "target_high_mg_dl": self.connection.target_high,
        }


class GlucoseLowSensor(_GlucoseThresholdBinarySensor):
    def __init__(
        self,
        coordinator: LibreViewCoordinator,
        connection_id: UUID,
        uom: GlucoseUnitOfMeasurement,
        max_data_age: int = DEFAULT_MAX_DATA_AGE,
    ):
        super().__init__(coordinator, connection_id, uom, max_data_age)
        self._attr_unique_id = f"{connection_id}_glucose_reading_is_low"

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        name = f"{self.connection.first_name } {self.connection.last_name}"
        return f"{name} glucose level is low"

    @property
    def is_on(self) -> bool:
        """Return the state of the entity."""
        return self.gcm.value_in_mg_per_dl < self.connection.target_low

    @property
    def extra_state_attributes(self) -> Dict[str, int | float]:
        return {
            "target_low_mmol_l": round(self.connection.target_low / 18, 1),
            "target_low_mg_dl": self.connection.target_low,
        }
