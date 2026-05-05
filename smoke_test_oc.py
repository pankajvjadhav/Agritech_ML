"""Smoke-test the OC + EC + nutrient calibration fixes against synthetic
SoilGrids fixtures. Exercises the live path that today's SoilGrids outage
prevents us from hitting end-to-end.

Run: ./venv/Scripts/python.exe smoke_test_oc.py
"""

from dotenv import load_dotenv
load_dotenv()

from app.data_fetcher import (
    _weighted_topsoil_mean,
    _estimate_ec,
    _OC_MIN_PCT,
    _OC_MAX_PCT,
    _SOIL_FALLBACK,
)
from app.predictor import hybrid_nitrogen_prediction, make_classification_prediction


# A representative SoilGrids `soc` layer for a fertile UP / Maharashtra
# loam: 12–18 g/kg in the topsoil — i.e. ~1.2–1.8 % OC. d_factor=10 is
# applied inside _weighted_topsoil_mean (raw integers come back as 120–180,
# divided by 10 → 12–18 g/kg).
soc_layer_realistic = {
    "name": "soc",
    "unit_measure": {"d_factor": 10, "mapped_units": "dg/kg", "target_units": "g/kg"},
    "depths": [
        {"label": "0-5cm", "values": {"mean": 180}},   # 18 g/kg
        {"label": "5-15cm", "values": {"mean": 160}},  # 16 g/kg
        {"label": "15-30cm", "values": {"mean": 130}}, # 13 g/kg
    ],
}

# Same shape but for a degraded sandy soil (~0.4 % OC — ICAR Low band).
soc_layer_low = {
    "name": "soc",
    "unit_measure": {"d_factor": 10, "mapped_units": "dg/kg", "target_units": "g/kg"},
    "depths": [
        {"label": "0-5cm", "values": {"mean": 50}},   # 5 g/kg → 0.5 % top
        {"label": "5-15cm", "values": {"mean": 40}},  # 4 g/kg
        {"label": "15-30cm", "values": {"mean": 30}}, # 3 g/kg
    ],
}

# An all-nulls layer like SoilGrids is currently serving — should fall over
# to the fallback in fetch_real_time_data.
soc_layer_nulls = {
    "name": "soc",
    "unit_measure": {"d_factor": 10},
    "depths": [
        {"label": "0-5cm", "values": {"mean": None}},
        {"label": "5-15cm", "values": {"mean": None}},
        {"label": "15-30cm", "values": {"mean": None}},
    ],
}


def assert_close(label, value, lo, hi):
    ok = (value is not None) and lo <= value <= hi
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: {value!r}  (expected {lo}–{hi})")
    return ok


print("=" * 72)
print("OC unit conversion (SoilGrids g/kg → ICAR %)")
print("=" * 72)
soc_g_kg = _weighted_topsoil_mean(soc_layer_realistic)
print(f"  Weighted SOC (0–30 cm): {soc_g_kg:.2f} g/kg")
oc_percent = soc_g_kg / 10.0
oc_clamped = max(_OC_MIN_PCT, min(_OC_MAX_PCT, oc_percent))
print(f"  OC after /10 conversion : {oc_percent:.2f} %")
print(f"  OC after ICAR clamp     : {oc_clamped:.2f} %")
assert_close("realistic loam OC", oc_clamped, 1.3, 1.6)
soc_g_kg_low = _weighted_topsoil_mean(soc_layer_low)
oc_low = max(_OC_MIN_PCT, min(_OC_MAX_PCT, soc_g_kg_low / 10.0))
print(f"  Sandy degraded soil OC  : {oc_low:.3f} %  ({soc_g_kg_low:.2f} g/kg)")
assert_close("degraded sandy OC", oc_low, 0.3, 0.5)

print()
print("=" * 72)
print("Fallback values are in ICAR-plausible ranges")
print("=" * 72)
print(f"  Fallback dict          : {_SOIL_FALLBACK}")
assert_close("fallback OC (% units)", _SOIL_FALLBACK["oc"], 0.1, 1.0)
assert_close("fallback EC (dS/m)", _SOIL_FALLBACK["ec"], 0.05, 1.0)
assert_close("fallback pH", _SOIL_FALLBACK["ph"], 5.5, 8.4)

print()
print("=" * 72)
print("EC surrogate stays inside ICAR salinity bands")
print("=" * 72)
ec_value, cat = _estimate_ec(
    soil_moisture=0.4, clay_content=25.0, oc_value=oc_clamped, rainfall_mean=5.0,
)
print(f"  Loam moisture=0.4 clay=25 oc={oc_clamped:.2f}% rain=5 → {ec_value} dS/m  ({cat})")
assert_close("EC realistic loam", ec_value, 0.15, 1.0)

ec_dry, cat_dry = _estimate_ec(
    soil_moisture=0.15, clay_content=10.0, oc_value=0.4, rainfall_mean=1.0,
)
print(f"  Sandy dry moisture=0.15 clay=10 oc=0.4 rain=1 → {ec_dry} dS/m  ({cat_dry})")
assert_close("EC sandy dry", ec_dry, 0.10, 0.6)

print()
print("=" * 72)
print("Nitrogen hybrid (post-fix calibration)")
print("=" * 72)
features = {
    "OC": 1.5, "NDVI_mean": 0.55, "NDMI_mean": 0.30, "rainfall_30d": 6.0, "NDVI_std": 0.05,
}
n_hybrid = hybrid_nitrogen_prediction(features)
# After the predictor's calibration: final * 1.05 + 30 (then clamped 60..700).
n_calibrated = max(60, min(700, n_hybrid * 1.05 + 30.0))
print(f"  Hybrid raw N            : {n_hybrid:.1f} kg/ha")
print(f"  Hybrid + calibration    : {n_calibrated:.1f} kg/ha")
assert_close("N within ICAR plausible band", n_calibrated, 100, 700)

# OC=0 case (the old bug — features had no `OC`/`oc` key)
n_zero_oc = hybrid_nitrogen_prediction({"NDVI_mean": 0.55, "NDMI_mean": 0.30,
                                        "rainfall_30d": 6.0, "NDVI_std": 0.05})
n_zero_calibrated = max(60, min(700, n_zero_oc * 1.05 + 30.0))
print(f"  Pre-fix bug (OC=0)      : N raw {n_zero_oc:.1f}  →  cal {n_zero_calibrated:.1f}")
print(f"  Net N gain from OC fix  : {n_calibrated - n_zero_calibrated:.1f} kg/ha")

print()
print("=" * 72)
print("End-to-end: make_classification_prediction with realistic features")
print("=" * 72)
e2e_features = {
    "NDVI_mean": 0.55, "NDMI_mean": 0.30, "BSI_mean": 0.10, "NBR2_mean": 0.10,
    "VV_mean": -10.0, "VH_mean": -16.0, "VV_VH_ratio_mean": 0.6,
    "elevation": 550.0, "slope": 1.2, "LST_mean": 32.0,
    "rainfall_sum": 180.0, "rainfall_mean": 6.0, "rainfall_max": 25.0, "rainy_days": 14,
    "soil_type": "Loamy", "month": 7,
    "ph": 6.7, "ec": 0.45, "oc": 1.5,
    "NDVI_rainfall": 3.3, "OC_rainfall": 9.0, "pH_OC": 10.05,
    "slope_rainfall": 7.2, "temp_moisture": 192.0, "NDVI_OC": 0.825, "NDMI_rain": 1.8,
}
result = make_classification_prediction(e2e_features)
for nut in ("nitrogen", "phosphorus", "potassium"):
    v = result[nut]["value"]
    s = result[nut]["status"]
    print(f"  {nut:12s} : {v:7.1f} kg/ha   [{s}]")
n_v = result["nitrogen"]["value"]; p_v = result["phosphorus"]["value"]; k_v = result["potassium"]["value"]
ok_n = assert_close("ICAR N plausible (60-700)", n_v, 60, 700)
ok_p = assert_close("ICAR P plausible (2-80)",  p_v, 2, 80)
ok_k = assert_close("ICAR K plausible (40-800)", k_v, 40, 800)

print()
print("=" * 72)
all_ok = all([ok_n, ok_p, ok_k])
print("RESULT:", "ALL GREEN" if all_ok else "SOME FAILURES")
print("=" * 72)
