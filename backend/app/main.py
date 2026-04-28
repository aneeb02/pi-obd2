import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes import telemetry
from backend.app.config import get_settings
from backend.app.services.csv_store import CsvStore
from backend.app.services.obd_reader import OBDReader
from backend.app.services.pid_scheduler import PidScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()
csv_store = CsvStore(data_dir=settings.data_dir)
reader = OBDReader(
    port=settings.obd_port,
    timeout_seconds=settings.obd_timeout_seconds,
    reconnect_base_delay_seconds=settings.reconnect_base_delay_seconds,
    reconnect_max_delay_seconds=settings.reconnect_max_delay_seconds,
    vehicle_session_id_factory=lambda: str(uuid.uuid4()),
)
scheduler = PidScheduler(reader=reader, csv_store=csv_store, interval_seconds=settings.poll_interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


def _get_scheduler() -> PidScheduler:
    return scheduler


import os
from fastapi.staticfiles import StaticFiles

app.include_router(telemetry.router)
app.dependency_overrides[telemetry.get_scheduler] = _get_scheduler

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")