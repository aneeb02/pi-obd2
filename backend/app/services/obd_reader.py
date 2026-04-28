import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

from backend.app.models.schemas import TelemetrySample

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PidDefinition:
    key: str
    name: str
    unit: str


DEFAULT_PIDS: tuple[PidDefinition, ...] = (
    PidDefinition("SPEED", "vehicle_speed", "km/h"),
    PidDefinition("RPM", "engine_rpm", "rpm"),
    PidDefinition("COOLANT_TEMP", "coolant_temp", "degC"),
    PidDefinition("THROTTLE_POS", "throttle_pos", "%"),
    PidDefinition("ENGINE_LOAD", "engine_load", "%"),
)


class OBDReader:
    """
    Reads PIDs from python-OBD when available.
    Falls back to simulation mode to support local development without hardware.
    """

    def __init__(
        self,
        *,
        port: str,
        timeout_seconds: float,
        reconnect_base_delay_seconds: float,
        reconnect_max_delay_seconds: float,
        vehicle_session_id_factory: Callable[[], str],
    ) -> None:
        self._port = port
        self._timeout_seconds = timeout_seconds
        self._reconnect_base_delay_seconds = reconnect_base_delay_seconds
        self._reconnect_max_delay_seconds = reconnect_max_delay_seconds
        self._vehicle_session_id_factory = vehicle_session_id_factory

        self._connection = None
        self._connected = False
        self._simulate = False
        self._lock = Lock()
        self._reconnect_count = 0
        self._poll_errors = 0
        self._session_id = vehicle_session_id_factory()

        self._connect()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def poll_errors(self) -> int:
        return self._poll_errors

    def _connect(self) -> None:
        try:
            import obd  # type: ignore

            self._connection = obd.OBD(portstr=self._port, timeout=self._timeout_seconds, fast=True)
            self._connected = bool(self._connection and self._connection.is_connected())
            if self._connected:
                self._simulate = False
                LOGGER.info("Connected to ELM327 on %s", self._port)
            else:
                LOGGER.warning("ELM327 not connected on %s", self._port)
        except Exception as exc:  # pylint: disable=broad-except
            self._connection = None
            self._connected = False
            self._simulate = True
            LOGGER.warning("Failed to initialize OBD adapter (%s), simulation mode enabled", exc)

    def _safe_reconnect(self) -> None:
        delay = self._reconnect_base_delay_seconds
        attempts = 0
        while not self._connected and attempts < 5:
            self._reconnect_count += 1
            attempts += 1
            LOGGER.warning("Attempting OBD reconnect #%s", self._reconnect_count)
            self._connect()
            if self._connected:
                return
            time.sleep(delay)
            delay = min(delay * 2, self._reconnect_max_delay_seconds)
        self._simulate = True
        LOGGER.warning("Reconnect attempts exhausted, simulation mode enabled")

    def _simulate_value(self, pid_key: str) -> float:
        if pid_key == "SPEED":
            return random.uniform(0, 120)
        if pid_key == "RPM":
            return random.uniform(700, 4000)
        if pid_key == "COOLANT_TEMP":
            return random.uniform(70, 105)
        if pid_key == "THROTTLE_POS":
            return random.uniform(0, 80)
        if pid_key == "ENGINE_LOAD":
            return random.uniform(10, 90)
        return 0.0

    def _query_pid_value(self, pid_key: str) -> float:
        if self._simulate:
            return self._simulate_value(pid_key)
        if not self._connected:
            self._safe_reconnect()
            if not self._connected:
                return self._simulate_value(pid_key)

        try:
            import obd  # type: ignore

            assert self._connection is not None
            command = getattr(obd.commands, pid_key)
            response = self._connection.query(command, force=True)
            if response is None or response.is_null():
                raise ValueError(f"Null response for {pid_key}")
            magnitude = getattr(response.value, "magnitude", None)
            if magnitude is None:
                return float(response.value)
            return float(magnitude)
        except Exception as exc:  # pylint: disable=broad-except
            self._poll_errors += 1
            LOGGER.warning("PID query failed for %s: %s", pid_key, exc)
            if self._connection is not None:
                self._connected = False
                self._safe_reconnect()
            return self._simulate_value(pid_key)

    def read_pids(self, pids: tuple[PidDefinition, ...] = DEFAULT_PIDS) -> list[TelemetrySample]:
        with self._lock:
            now = datetime.now(tz=timezone.utc)
            samples: list[TelemetrySample] = []
            for pid in pids:
                value = self._query_pid_value(pid.key)
                samples.append(
                    TelemetrySample(
                        timestamp=now,
                        vehicle_session_id=self._session_id,
                        pid=pid.key,
                        name=pid.name,
                        value=round(value, 2),
                        unit=pid.unit,
                    )
                )
            return samples
