from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = Field(default="pi-obd2-api", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    obd_port: str = Field(default="/dev/rfcomm0", alias="OBD_PORT")
    obd_baudrate: int = Field(default=38400, alias="OBD_BAUDRATE")
    obd_timeout_seconds: float = Field(default=2.0, alias="OBD_TIMEOUT_SECONDS")
    poll_interval_seconds: float = Field(default=1.0, alias="POLL_INTERVAL_SECONDS")
    reconnect_base_delay_seconds: float = Field(default=1.0, alias="RECONNECT_BASE_DELAY_SECONDS")
    reconnect_max_delay_seconds: float = Field(default=30.0, alias="RECONNECT_MAX_DELAY_SECONDS")

    data_dir: Path = Field(default=Path("/home/dev/pi-obd2/data"), alias="DATA_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()
