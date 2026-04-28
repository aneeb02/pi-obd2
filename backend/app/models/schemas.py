from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TelemetrySample(BaseModel):
    timestamp: datetime
    vehicle_session_id: str
    pid: str
    name: str
    value: float
    unit: str


class LatestTelemetryResponse(BaseModel):
    samples: list[TelemetrySample]


class HealthResponse(BaseModel):
    status: str
    connected: bool
    reconnect_count: int
    poll_errors: int
    uptime_seconds: float


class TelemetryRangeQuery(BaseModel):
    from_ts: Optional[datetime] = None
    to_ts: Optional[datetime] = None
    pid: Optional[str] = None
