from datetime import datetime, timezone

from backend.app.models.schemas import TelemetrySample
from backend.app.services.csv_store import CsvStore


def test_csv_store_append_and_read_range(tmp_path):
    store = CsvStore(data_dir=tmp_path)
    ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sample = TelemetrySample(
        timestamp=ts,
        vehicle_session_id="session-1",
        pid="RPM",
        name="engine_rpm",
        value=1234.5,
        unit="rpm",
    )

    store.append(sample)
    rows = store.read_range(pid="RPM")

    assert len(rows) == 1
    assert rows[0].pid == "RPM"
    assert rows[0].value == 1234.5
