import csv
import time
from datetime import datetime, timedelta

from satellite_features.sentinel2_features import get_sentinel2_features
from satellite_features.sentinel1_features import get_sentinel1_features
from satellite_features.terrain_features import get_terrain_features
from satellite_features.landsat_features import get_landsat_features
from satellite_features.rainfall_features import get_rainfall_features
from satellite_features.soil_type_features import get_soil_type

# 🔹 File paths
input_file = "data/soil_lab_data.csv"
output_file = "data/final_training_dataset.csv"

# 🔹 Fixed column order (VERY IMPORTANT)
fieldnames = [
    # Satellite features
    "NDVI_mean", "NDMI_mean", "BSI_mean", "NBR2_mean",
    "VV_mean", "VH_mean", "VV_VH_ratio_mean",
    "elevation", "slope",
    "LST_mean",
    "rainfall_sum", "rainfall_mean", "rainfall_max", "rainy_days",
    "soil_type",

    # Time
    "month",

    # Soil properties (Stage 1)
    "ph", "ec", "oc",

    # Nutrients (Stage 2)
    "nitrogen", "phosphorus", "potassium",
    "iron", "manganese", "zinc", "copper"
]

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    print("🚀 Generating training dataset...\n")

    for idx, row in enumerate(reader):
        try:
            # 🔹 Coordinates
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            # 🔹 Date handling
            sample_date = datetime.strptime(row["date"], "%Y-%m")

            start_date = (sample_date - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = (sample_date + timedelta(days=30)).strftime("%Y-%m-%d")

            month = sample_date.month

            # 🔹 Feature extraction
            s2 = get_sentinel2_features(lat, lon, start_date, end_date)
            s1 = get_sentinel1_features(lat, lon, start_date, end_date)
            terrain = get_terrain_features(lat, lon)
            landsat = get_landsat_features(lat, lon, start_date, end_date)
            rainfall = get_rainfall_features(lat, lon, start_date, end_date)
            soil = get_soil_type(lat, lon)

            # 🔹 Combine everything
            combined = {
                **s2,
                **s1,
                **terrain,
                **landsat,
                **rainfall,
                **soil,

                "month": month,

                "ph": float(row["ph"]),
                "ec": float(row["ec"]),
                "oc": float(row["oc"]),

                "nitrogen": float(row["nitrogen"]),
                "phosphorus": float(row["phosphorus"]),
                "potassium": float(row["potassium"]),
                "iron": float(row["iron"]),
                "manganese": float(row["manganese"]),
                "zinc": float(row["zinc"]),
                "copper": float(row["copper"]),
            }

            # 🔹 Handle missing values safely
            if any(v is None for v in combined.values()):
                print(f"⚠️ Skipping row {idx+1} due to missing values")
                continue

            # 🔹 Ensure all columns exist (IMPORTANT)
            for key in fieldnames:
                if key not in combined:
                    combined[key] = 0

            writer.writerow(combined)

            print(f"✅ Done {idx+1}: ({lat}, {lon})")

            # 🔹 Prevent Earth Engine rate limit
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error at row {idx+1}: {e}")
            continue

print("\n🎉 Training dataset generated successfully!")
print(f"📁 Saved at: {output_file}")