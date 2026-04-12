import ee

# Initialize Earth Engine safely
try:
    ee.Initialize(project='soil-model-490412')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='soil-model-490412')

def get_landsat_features(lat, lon, start_date="2025-05-01", end_date="2025-05-31"):
    try:
        point = ee.Geometry.Point([lon, lat])

        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .median()
        )

        # 🌡 Land Surface Temperature
        lst = collection.select("ST_B10") \
            .multiply(0.00341802) \
            .add(149.0) \
            .rename("LST")

        values = lst.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        )

        data = values.getInfo()

        features = {
            "LST_mean": data.get("LST", 0)
        }

        return features

    except Exception as e:
        print(f"Landsat Error: {e}")

        return {
            "LST_mean": 0
        }