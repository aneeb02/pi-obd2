"""
SQLAlchemy database models for the Pi OBD2 Telemetry system.

Uses SQLite for lightweight, serverless persistence on the Raspberry Pi.
Tables mirror the CSV data structure but add relational integrity
and efficient querying via indexes.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class VehicleSession(Base):
    """A driving session — starts when the Pi boots / connects to OBD-II."""
    __tablename__ = "vehicle_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_uuid = Column(String(36), unique=True, nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    reading_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    readings = relationship("TelemetryReading", back_populates="session", cascade="all, delete-orphan")
    reports = relationship("DiagnosticReport", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<VehicleSession {self.session_uuid[:8]}… readings={self.reading_count}>"


class TelemetryReading(Base):
    """A single PID reading at a point in time."""
    __tablename__ = "telemetry_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("vehicle_sessions.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    pid = Column(String(32), nullable=False, index=True)
    name = Column(String(64), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(16), nullable=False)

    session = relationship("VehicleSession", back_populates="readings")

    def __repr__(self) -> str:
        return f"<TelemetryReading {self.name}={self.value}{self.unit}>"


class DiagnosticReport(Base):
    """ML anomaly detection results for a session."""
    __tablename__ = "diagnostic_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("vehicle_sessions.id"), nullable=False, index=True)
    anomaly_score = Column(Float, nullable=True)
    severity = Column(String(16), nullable=True)  # LOW / MEDIUM / HIGH
    summary = Column(Text, nullable=True)
    model_used = Column(String(64), default="isolation_forest")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("VehicleSession", back_populates="reports")

    def __repr__(self) -> str:
        return f"<DiagnosticReport session={self.session_id} severity={self.severity}>"


# ── Database engine factory ──────────────────────────────────────────

def get_engine(db_path: str = "data/telemetry.db"):
    """Create a SQLAlchemy engine for the SQLite database."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str = "data/telemetry.db"):
    """Create all tables if they don't exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine):
    """Return a session factory bound to the given engine."""
    return sessionmaker(bind=engine)
