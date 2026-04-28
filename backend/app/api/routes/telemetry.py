import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from backend.app.models.schemas import HealthResponse, LatestTelemetryResponse, TelemetrySample
from backend.app.services.pid_scheduler import PidScheduler

router = APIRouter()

def get_scheduler() -> PidScheduler:
    raise RuntimeError("Scheduler dependency not wired")

@router.get("/health", response_model=HealthResponse)
async def health(scheduler: PidScheduler = Depends(get_scheduler)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        connected=scheduler.connected,
        reconnect_count=scheduler.reconnect_count,
        poll_errors=scheduler.poll_errors,
        uptime_seconds=round(scheduler.uptime_seconds, 2),
    )


@router.get("/telemetry/latest", response_model=LatestTelemetryResponse)
async def telemetry_latest(scheduler: PidScheduler = Depends(get_scheduler)) -> LatestTelemetryResponse:
    return LatestTelemetryResponse(samples=scheduler.latest_samples())


@router.get("/telemetry/range", response_model=list[TelemetrySample])
async def telemetry_range(
    scheduler: PidScheduler = Depends(get_scheduler),
    from_ts: datetime | None = Query(default=None),
    to_ts: datetime | None = Query(default=None),
    pid: str | None = Query(default=None),
    pid_upper: str | None = Query(default=None, alias="PID"),
) -> list[TelemetrySample]:
    return scheduler.read_range(from_ts=from_ts, to_ts=to_ts, pid=pid or pid_upper)


@router.get("/telemetry/plot", response_class=HTMLResponse)
async def telemetry_plot(
    scheduler: PidScheduler = Depends(get_scheduler),
    pid: str | None = Query(default=None),
    pid_upper: str | None = Query(default=None, alias="PID"),
    from_ts: datetime | None = Query(default=None),
    to_ts: datetime | None = Query(default=None),
) -> str:
    resolved_pid = (pid or pid_upper or "SPEED").upper()
    samples = scheduler.read_range(from_ts=from_ts, to_ts=to_ts, pid=resolved_pid)
    labels = [sample.timestamp.isoformat() for sample in samples]
    values = [sample.value for sample in samples]
    pid_label = resolved_pid

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telemetry Plot - {pid_label}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body {{ font-family: sans-serif; margin: 24px; }}
      h2 {{ margin-bottom: 8px; }}
      .meta {{ color: #555; margin-bottom: 18px; }}
    </style>
  </head>
  <body>
    <h2>Telemetry Plot: {pid_label}</h2>
    <div class="meta">Samples: {len(values)}</div>
    <canvas id="chart" height="110"></canvas>
    <script>
      const labels = {labels};
      const values = {values};
      new Chart(document.getElementById('chart'), {{
        type: 'line',
        data: {{
          labels,
          datasets: [{{
            label: '{pid_label}',
            data: values,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2
          }}]
        }},
        options: {{
          responsive: true,
          scales: {{
            x: {{ ticks: {{ maxTicksLimit: 10 }} }},
            y: {{ beginAtZero: false }}
          }}
        }}
      }});
    </script>
  </body>
</html>
"""


@router.websocket("/telemetry/stream")
async def telemetry_stream(websocket: WebSocket, scheduler: PidScheduler = Depends(get_scheduler)) -> None:
    await websocket.accept()
    queue: asyncio.Queue[TelemetrySample] = asyncio.Queue()

    async def consumer(sample: TelemetrySample) -> None:
        await queue.put(sample)

    await scheduler.subscribe(consumer)
    try:
        while True:
            sample = await queue.get()
            await websocket.send_text(sample.model_dump_json())
    except WebSocketDisconnect:
        return
    finally:
        await scheduler.unsubscribe(consumer)
