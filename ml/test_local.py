import numpy as np
import pandas as pd
from inference import OverheatArtifacts, predict_overheat_from_readings
from preprocess import BACKBONE_FEATURES
def generate_dummy_readings(rows=70, overheating=False):
    np.random.seed(42)

    # Base normal behavior
    coolant = 88 + np.random.normal(0, 0.5, rows)
    rpm = 1500 + np.random.normal(0, 80, rows)
    map_kpa = 35 + np.random.normal(0, 2, rows)
    maf = 9 + np.random.normal(0, 0.3, rows)
    throttle = 20 + np.random.normal(0, 2, rows)
    iat = 32 + np.random.normal(0, 0.5, rows)
    ambient = 30 + np.random.normal(0, 0.2, rows)

    if overheating:
        # Simulate rising coolant + load
        coolant[40:] += np.linspace(5, 15, rows - 40)
        rpm[40:] += 300
        map_kpa[40:] += 8
        maf[40:] += 2
        throttle[40:] += 5

    df = pd.DataFrame({
        "Engine Coolant Temperature [°C]": coolant,
        "Intake Manifold Absolute Pressure [kPa]": map_kpa,
        "Engine RPM [RPM]": rpm,
        "Intake Air Temperature [°C]": iat,
        "Air Flow Rate from Mass Flow Sensor [g/s]": maf,
        "Absolute Throttle Position [%]": throttle,
        "Ambient Air Temperature [°C]": ambient,
    })

    return df.to_dict(orient="records")

artifacts = OverheatArtifacts("lstm_artifacts")
print("NORMAL:\n")
print(predict_overheat_from_readings(
    generate_dummy_readings(70, overheating=False),
    artifacts
))
print("\n" + "="*50 + "\n")
print("\nOVERHEAT:")
print(predict_overheat_from_readings(
    generate_dummy_readings(70, overheating=True),
    artifacts
))