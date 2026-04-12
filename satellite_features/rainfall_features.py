import ee

from satellite_features.gee_client import initialize_earth_engine


initialize_earth_engine()

def get_rainfall_features(lat, lon, start_date, end_date):
    try:
        point = ee.Geometry.Point([lon, lat])

        # 🌧️ CHIRPS Rainfall Dataset
        rainfall_collection = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(point)
            .filterDate(start_date, end_date)
        )

        # 🔥 Features
        rainfall_sum = rainfall_collection.sum().rename("rainfall_sum")
        rainfall_mean = rainfall_collection.mean().rename("rainfall_mean")
        rainfall_max = rainfall_collection.max().rename("rainfall_max")

        rainy_days = rainfall_collection.map(
            lambda img: img.gt(1)
        ).sum().rename("rainy_days")

        final_image = ee.Image.cat([
            rainfall_sum,
            rainfall_mean,
            rainfall_max,
            rainy_days
        ])

        values = final_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=5000,
            maxPixels=1e9
        )

        data = values.getInfo()

        features = {
            "rainfall_sum": data.get("rainfall_sum", 0),
            "rainfall_mean": data.get("rainfall_mean", 0),
            "rainfall_max": data.get("rainfall_max", 0),
            "rainy_days": data.get("rainy_days", 0),
        }

        return features

    except Exception as e:
        print(f"Rainfall Error: {e}")

        raise