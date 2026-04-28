import asyncio
from datetime import datetime, timezone

from backend.app.models.schemas import TelemetrySample
from backend.app.services.csv_store import CsvStore
from backend.app.services.pid_scheduler import PidScheduler


class FakeReader:
    connected = True
    reconnect_count = 0
    poll_errors = 0

    def read_pids(self):
        return [
            TelemetrySample(
                timestamp=datetime.now(tz=timezone.utc),
                vehicle_session_id="session-1",
                pid="SPEED",
                name="vehicle_speed",
                value=42.0,
                unit="km/h",
            )
        ]


def test_scheduler_updates_latest_and_persists(tmp_path):
    scheduler = PidScheduler(reader=FakeReader(), csv_store=CsvStore(tmp_path), interval_seconds=0.01)

    async def run_once():
        task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_once())

    latest = scheduler.latest_samples()
    assert latest
    assert latest[0].pid == "SPEED"
    assert scheduler.read_range(pid="SPEED")
