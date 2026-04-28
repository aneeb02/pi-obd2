import csv
from datetime import datetime
from pathlib import Path
from threading import Lock

from backend.app.models.schemas import TelemetrySample

CSV_HEADERS = ["timestamp", "vehicle_session_id", "pid", "name", "value", "unit"]


class CsvStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._lock = Lock()
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path_for_day(self, sample_time: datetime) -> Path:
        return self._data_dir / f"{sample_time.date().isoformat()}.csv"

    def append(self, sample: TelemetrySample) -> None:
        path = self._csv_path_for_day(sample.timestamp)
        with self._lock:
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": sample.timestamp.isoformat(),
                        "vehicle_session_id": sample.vehicle_session_id,
                        "pid": sample.pid,
                        "name": sample.name,
                        "value": sample.value,
                        "unit": sample.unit,
                    }
                )

    def read_range(
        self, *, from_ts: datetime | None = None, to_ts: datetime | None = None, pid: str | None = None
    ) -> list[TelemetrySample]:
        rows: list[TelemetrySample] = []
        normalized_pid = pid.strip().upper() if pid else None
        for csv_path in sorted(self._data_dir.glob("*.csv")):
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample = TelemetrySample(
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        vehicle_session_id=row["vehicle_session_id"],
                        pid=row["pid"],
                        name=row["name"],
                        value=float(row["value"]),
                        unit=row["unit"],
                    )
                    if from_ts and sample.timestamp < from_ts:
                        continue
                    if to_ts and sample.timestamp > to_ts:
                        continue
                    if normalized_pid and sample.pid.upper() != normalized_pid:
                        continue
                    rows.append(sample)
        return rows
