import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from backend.app.models.schemas import TelemetrySample
from backend.app.services.csv_store import CsvStore
from backend.app.services.obd_reader import OBDReader

LOGGER = logging.getLogger(__name__)

SampleConsumer = Callable[[TelemetrySample], Awaitable[None]]


class PidScheduler:
    def __init__(self, *, reader: OBDReader, csv_store: CsvStore, interval_seconds: float) -> None:
        self._reader = reader
        self._csv_store = csv_store
        self._interval_seconds = interval_seconds
        self._latest_by_pid: dict[str, TelemetrySample] = {}
        self._subscribers: set[SampleConsumer] = set()
        self._task: asyncio.Task | None = None
        self._started_at_monotonic = time.monotonic()

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started_at_monotonic

    @property
    def connected(self) -> bool:
        return self._reader.connected

    @property
    def reconnect_count(self) -> int:
        return self._reader.reconnect_count

    @property
    def poll_errors(self) -> int:
        return self._reader.poll_errors

    def latest_samples(self) -> list[TelemetrySample]:
        return sorted(self._latest_by_pid.values(), key=lambda sample: sample.pid)

    def read_range(
        self, *, from_ts: datetime | None = None, to_ts: datetime | None = None, pid: str | None = None
    ) -> list[TelemetrySample]:
        return self._csv_store.read_range(from_ts=from_ts, to_ts=to_ts, pid=pid)

    async def subscribe(self, callback: SampleConsumer) -> None:
        self._subscribers.add(callback)

    async def unsubscribe(self, callback: SampleConsumer) -> None:
        self._subscribers.discard(callback)

    async def _broadcast(self, sample: TelemetrySample) -> None:
        if not self._subscribers:
            return
        failed: list[SampleConsumer] = []
        for callback in self._subscribers:
            try:
                await callback(sample)
            except Exception:  # pylint: disable=broad-except
                failed.append(callback)
        for callback in failed:
            self._subscribers.discard(callback)

    async def run(self) -> None:
        LOGGER.info("PID scheduler started with interval=%ss", self._interval_seconds)
        while True:
            samples = self._reader.read_pids()
            for sample in samples:
                self._latest_by_pid[sample.pid] = sample
                self._csv_store.append(sample)
                await self._broadcast(sample)
            await asyncio.sleep(self._interval_seconds)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            LOGGER.info("PID scheduler cancelled")
