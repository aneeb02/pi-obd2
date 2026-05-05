import numpy as np
import pandas as pd

BACKBONE_FEATURES = [
    'Engine Coolant Temperature [°C]',
    'Intake Manifold Absolute Pressure [kPa]',
    'Engine RPM [RPM]',
    'Intake Air Temperature [°C]',
    'Air Flow Rate from Mass Flow Sensor [g/s]',
    'Absolute Throttle Position [%]',
    'Ambient Air Temperature [°C]'
]
def validate_input_df(df: pd.DataFrame, features=BACKBONE_FEATURES, min_rows: int = 50) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    if len(df) < min_rows:
        raise ValueError(f"Not enough timesteps. Need at least {min_rows}, got {len(df)}")


def preprocess_obd_df(df: pd.DataFrame, features=BACKBONE_FEATURES) -> pd.DataFrame:
    """
    Matches your training behavior:
    - select columns in correct order
    - numeric conversion
    - replace 0 -> NaN
    - interpolate + ffill + bfill
    """
    df = df[features].copy()

    # Ensure numeric
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").astype(float)

    # Handle zero spikes and missing values
    df = df.replace(0, np.nan)
    df = df.interpolate(method="linear").ffill().bfill()

    return df
def scale_obd(df_clean: pd.DataFrame, scaler) -> np.ndarray:
   #already saved scaler thats been fitted on training data, so we can just transform the new data
    return scaler.transform(df_clean.values)
def create_sequences(data: np.ndarray, seq_len: int = 50, hop: int = 1) -> np.ndarray:
    """
    Sliding windows over scaled data.
    data shape: (N, 7)
    returns: (num_windows, seq_len, 7)
    """
    if hop <= 0:
        hop = 1

    n = len(data)
    if n < seq_len:
        return np.empty((0, seq_len, data.shape[1]), dtype=np.float32)

    seqs = []
    for start in range(0, n - seq_len + 1, hop):
        seqs.append(data[start:start + seq_len])

    return np.asarray(seqs, dtype=np.float32)