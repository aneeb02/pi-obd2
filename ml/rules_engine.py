from typing import Any, Dict, List
import pandas as pd 
def _add_reason(
    reasons: List[Dict[str, Any]],
    rule_id: str,
    title: str,
    severity: str,
    observation: str,
    why_it_matters: str,
    likely_causes: List[str],
    checks_next: List[str],
    safety_advice: List[str],
    supporting_metrics: Dict[str, float],
):
    reasons.append({
        "rule_id": rule_id,
        "title": title,
        "severity": severity,  # LOW | MEDIUM | HIGH | CRITICAL
        "observation": observation,
        "why_it_matters": why_it_matters,
        "likely_causes": likely_causes,
        "checks_next": checks_next,
        "safety_advice": safety_advice,
        "supporting_metrics": supporting_metrics,
    })


def generate_overheat_reasons(df_clean: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns structured reasons suitable for LLM explanation generation.
    Uses only the available 7 OBD features (no IoT yet).
    """
    reasons: List[Dict[str, Any]] = []

    # ---- Core stats ----
    ECT_series = df_clean["Engine Coolant Temperature [°C]"]
    AAT_series = df_clean["Ambient Air Temperature [°C]"]
    RPM_series = df_clean["Engine RPM [RPM]"]
    THR_series = df_clean["Absolute Throttle Position [%]"]
    IAT_series = df_clean["Intake Air Temperature [°C]"]
    MAP_series = df_clean["Intake Manifold Absolute Pressure [kPa]"]
    MAF_series = df_clean["Air Flow Rate from Mass Flow Sensor [g/s]"]

    ECT_mean = float(ECT_series.mean())
    ECT_max = float(ECT_series.max())
    ECT_min = float(ECT_series.min())
    # “Rise” over the window (simple, works for short scans)
    ECT_delta = float(ECT_series.iloc[-1] - ECT_series.iloc[0])

    AAT_mean = float(AAT_series.mean())
    RPM_mean = float(RPM_series.mean())
    THR_mean = float(THR_series.mean())
    IAT_mean = float(IAT_series.mean())
    MAP_mean = float(MAP_series.mean())
    MAF_mean = float(MAF_series.mean())

    dT_mean = float(ECT_mean - AAT_mean)  # coolant vs ambient

    # Add a small “context block” reason for the LLM (optional but useful)
    _add_reason(
        reasons=reasons,
        rule_id="CTX_WINDOW_SUMMARY",
        title="Window summary",
        severity="LOW",
        observation="Summary statistics computed from the scan window to support diagnosis.",
        why_it_matters="These metrics provide context for interpreting overheating risk and related engine-load behavior.",
        likely_causes=[],
        checks_next=[],
        safety_advice=[],
        supporting_metrics={
            "ECT_mean_C": ECT_mean,
            "ECT_max_C": ECT_max,
            "ECT_min_C": ECT_min,
            "ECT_delta_C": ECT_delta,
            "AAT_mean_C": AAT_mean,
            "dT_mean_C": dT_mean,
            "RPM_mean": RPM_mean,
            "THR_mean_pct": THR_mean,
            "IAT_mean_C": IAT_mean,
            "MAP_mean_kPa": MAP_mean,
            "MAF_mean_gps": MAF_mean,
        },
    )

    # ---- Rule 1: High coolant temperature (absolute) ----
    if ECT_max >= 110 or ECT_mean >= 105:
        sev = "CRITICAL"
    elif ECT_max >= 105 or ECT_mean >= 100:
        sev = "HIGH"
    elif ECT_max >= 100:
        sev = "MEDIUM"
    else:
        sev = ""

    if sev:
        _add_reason(
            reasons,
            rule_id="R1_HIGH_COOLANT_TEMP",
            title="Coolant temperature is high",
            severity=sev,
            observation=(
                f"Coolant temperature is elevated during this window "
                f"(mean={ECT_mean:.1f}°C, max={ECT_max:.1f}°C)."
            ),
            why_it_matters=(
                "Sustained high coolant temperature can indicate insufficient heat removal. "
                "If temperature continues to rise, it may lead to engine damage (head gasket, warped head) "
                "or forced limp mode."
            ),
            likely_causes=[
                "Low coolant level / coolant leak",
                "Radiator fan not engaging (especially at low speed/idle)",
                "Thermostat stuck closed or partially blocked",
                "Radiator blockage / poor airflow through radiator fins",
                "Water pump inefficiency or belt/impeller issues",
            ],
            checks_next=[
                "Check coolant reservoir level (engine cool) and look for visible leaks",
                "Confirm radiator fan turns on when engine warms up",
                "Inspect radiator fins for blockage (dust, debris)",
                "If safe: monitor whether temperature drops when vehicle speed increases (airflow effect)",
            ],
            safety_advice=[
                "If dashboard temperature warning light appears, pull over safely and stop the engine.",
                "Do not open radiator cap when hot (burn risk). Wait for cooling before inspection.",
            ],
            supporting_metrics={
                "ECT_mean_C": ECT_mean,
                "ECT_max_C": ECT_max,
            },
        )

    # ---- Rule 2: Rapid coolant rise (trend) ----
    # For a 50-step window, even +5°C can be meaningful depending on time scale
    if ECT_delta >= 10:
        sev = "HIGH"
    elif ECT_delta >= 6:
        sev = "MEDIUM"
    elif ECT_delta >= 3:
        sev = "LOW"
    else:
        sev = ""

    if sev:
        _add_reason(
            reasons,
            rule_id="R2_RISING_COOLANT_TREND",
            title="Coolant temperature is rising quickly",
            severity=sev,
            observation=f"Coolant increased by {ECT_delta:.1f}°C over the scan window.",
            why_it_matters=(
                "A fast upward trend suggests the cooling system may be failing to stabilize temperature. "
                "This is often more concerning than a single high reading, because it indicates the system "
                "is not reaching equilibrium."
            ),
            likely_causes=[
                "Cooling fan not turning on during idle/low speed",
                "Thermostat delayed opening or stuck",
                "Air trapped in cooling system after recent refill",
                "Radiator flow restriction or clogged passages",
            ],
            checks_next=[
                "Repeat scan after 1–2 minutes to confirm trend continues",
                "Check whether RPM/Throttle changed significantly (load-driven heating)",
                "If possible, compare ECT behavior at idle vs moving speed",
            ],
            safety_advice=[
                "If the trend continues upward, avoid heavy throttle and stop if warning appears.",
            ],
            supporting_metrics={
                "ECT_delta_C": ECT_delta,
                "ECT_start_C": float(ECT_series.iloc[0]),
                "ECT_end_C": float(ECT_series.iloc[-1]),
            },
        )

    # ---- Rule 3: Large delta between coolant and ambient ----
    # This is a heuristic; ambient varies by region; you can tune thresholds later.
    if dT_mean >= 80:
        sev = "HIGH"
    elif dT_mean >= 70:
        sev = "MEDIUM"
    elif dT_mean >= 60:
        sev = "LOW"
    else:
        sev = ""

    if sev:
        _add_reason(
            reasons,
            rule_id="R3_HIGH_COOLANT_AMBIENT_DELTA",
            title="Coolant-to-ambient temperature delta is large",
            severity=sev,
            observation=f"Average coolant-ambient delta is {dT_mean:.1f}°C (ECT_mean={ECT_mean:.1f}°C, AAT_mean={AAT_mean:.1f}°C).",
            why_it_matters=(
                "A large delta indicates the engine is much hotter than surrounding air. "
                "In very hot weather the delta naturally reduces cooling efficiency; however an unusually "
                "large delta can also suggest cooling system inefficiency or sustained heat load."
            ),
            likely_causes=[
                "Radiator airflow restriction / fan issue",
                "Insufficient coolant flow",
                "High engine load over time",
            ],
            checks_next=[
                "Check vehicle speed during scan (overheating at idle suggests fan issue)",
                "Inspect radiator/AC condenser area for blockage",
            ],
            safety_advice=[
                "In hot weather, avoid prolonged idling if overheating symptoms appear.",
            ],
            supporting_metrics={
                "dT_mean_C": dT_mean,
                "ECT_mean_C": ECT_mean,
                "AAT_mean_C": AAT_mean,
            },
        )

    # ---- Rule 4: High ECT at idle/low RPM ----
    if ECT_mean >= 100 and RPM_mean < 1000:
        _add_reason(
            reasons,
            rule_id="R4_OVERHEAT_AT_IDLE",
            title="Overheating pattern at idle / low RPM",
            severity="HIGH",
            observation=f"Coolant is high while RPM is low (ECT_mean={ECT_mean:.1f}°C, RPM_mean={RPM_mean:.0f}).",
            why_it_matters=(
                "Overheating at idle often points to inadequate airflow through the radiator because "
                "vehicle motion is low. This frequently implicates the radiator fan or fan control circuit."
            ),
            likely_causes=[
                "Radiator fan not engaging or weak fan",
                "Fan relay / fuse / wiring fault",
                "Cooling fan control issue",
                "Partially blocked radiator (low airflow impact)"
            ],
            checks_next=[
                "Confirm radiator fan turns on when engine warms up",
                "Check fuses/relays (if accessible)",
                "Observe whether temperature drops when driving at moderate speed",
            ],
            safety_advice=[
                "If temperature rises quickly at idle, avoid staying stationary; stop engine if warning appears."
            ],
            supporting_metrics={
                "ECT_mean_C": ECT_mean,
                "RPM_mean": RPM_mean,
            },
        )

    # ---- Rule 5: Overheating under low throttle (load-independent) ----
    if ECT_mean >= 100 and THR_mean < 10:
        _add_reason(
            reasons,
            rule_id="R5_OVERHEAT_LOW_THROTTLE",
            title="High coolant temperature under low throttle",
            severity="MEDIUM",
            observation=f"Coolant temperature is high while throttle is low (ECT_mean={ECT_mean:.1f}°C, THR_mean={THR_mean:.1f}%).",
            why_it_matters=(
                "If the engine is not being driven hard but temperature is still high, "
                "this suggests a cooling system limitation rather than purely load-driven heating."
            ),
            likely_causes=[
                "Thermostat partially stuck",
                "Coolant flow restriction / blockage",
                "Low coolant level",
                "Radiator efficiency reduced"
            ],
            checks_next=[
                "Re-check coolant level and inspect for leaks",
                "Repeat scan under the same low-load condition to confirm",
            ],
            safety_advice=[
                "Avoid pushing the engine until the cause is identified if overheating repeats."
            ],
            supporting_metrics={
                "ECT_mean_C": ECT_mean,
                "THR_mean_pct": THR_mean,
            },
        )

    # ---- Rule 6: Intake air much hotter than ambient ----
    iat_delta = float(IAT_mean - AAT_mean)
    if iat_delta >= 25:
        sev = "MEDIUM"
    elif iat_delta >= 15:
        sev = "LOW"
    else:
        sev = ""

    if sev:
        _add_reason(
            reasons,
            rule_id="R6_IAT_HIGH_VS_AMBIENT",
            title="Intake air temperature is high relative to ambient",
            severity=sev,
            observation=f"IAT is {iat_delta:.1f}°C higher than ambient (IAT_mean={IAT_mean:.1f}°C, AAT_mean={AAT_mean:.1f}°C).",
            why_it_matters=(
                "Hot intake air can indicate heat soak or reduced airflow. While not a direct overheating proof, "
                "it supports a scenario where under-hood heat is elevated or airflow is restricted."
            ),
            likely_causes=[
                "Heat soak (especially after idling)",
                "Restricted intake airflow / clogged filter",
                "Poor airflow through engine bay"
            ],
            checks_next=[
                "Check air filter condition (if accessible)",
                "Repeat scan after driving (heat soak should reduce with airflow)",
            ],
            safety_advice=[],
            supporting_metrics={
                "IAT_mean_C": IAT_mean,
                "AAT_mean_C": AAT_mean,
                "IAT_minus_AAT_C": iat_delta,
            },
        )

    # ---- Rule 7: MAP/MAF sanity heuristics (vehicle-dependent) ----
    # These are weaker signals; keep severity low unless extreme.
    if MAP_mean < 20:
        _add_reason(
            reasons,
            rule_id="R7_MAP_VERY_LOW",
            title="MAP is unusually low (vehicle-dependent)",
            severity="LOW",
            observation=f"MAP_mean={MAP_mean:.1f} kPa appears low for typical conditions.",
            why_it_matters=(
                "Very low MAP may indicate high vacuum (closed throttle) or a sensor/data issue. "
                "If paired with overheating, it can help interpret whether the engine was under load."
            ),
            likely_causes=[
                "Closed throttle / deceleration condition",
                "MAP sensor reading issue",
                "Vacuum leak (symptoms vary)"
            ],
            checks_next=[
                "Compare MAP with throttle and RPM at the same time window",
                "Verify PID parsing and units are correct",
            ],
            safety_advice=[],
            supporting_metrics={"MAP_mean_kPa": MAP_mean},
        )

    if MAF_mean < 2:
        _add_reason(
            reasons,
            rule_id="R8_MAF_VERY_LOW",
            title="MAF is unusually low (vehicle-dependent)",
            severity="LOW",
            observation=f"MAF_mean={MAF_mean:.2f} g/s appears low.",
            why_it_matters=(
                "Low airflow could be normal at idle, but can also indicate airflow restriction or sensor drift. "
                "Not a direct overheating indicator, but useful context."
            ),
            likely_causes=[
                "Normal idle airflow (may be normal)",
                "Dirty air filter",
                "MAF sensor contamination"
            ],
            checks_next=[
                "Check whether RPM was idle-level; if yes, low MAF may be expected",
                "If not idle, inspect intake/MAF"
            ],
            safety_advice=[],
            supporting_metrics={"MAF_mean_gps": MAF_mean, "RPM_mean": RPM_mean},
        )

    # If only the context block exists, add a “no issues” summary for the LLM
    if len(reasons) == 1 and reasons[0]["rule_id"] == "CTX_WINDOW_SUMMARY":
        _add_reason(
            reasons,
            rule_id="R0_NO_RULE_VIOLATIONS",
            title="No overheating indicators detected by rules",
            severity="LOW",
            observation="None of the heuristic overheating rules were triggered for this scan window.",
            why_it_matters=(
                "This suggests temperature and supporting signals are within a typical operating range for this window. "
                "ML anomaly score should be considered together with these rules."
            ),
            likely_causes=[],
            checks_next=[
                "If symptoms exist (steam, smell, warning light), re-scan and inspect cooling system manually."
            ],
            safety_advice=[],
            supporting_metrics={},
        )

    return reasons