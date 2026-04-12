import ee

from satellite_features.gee_client import initialize_earth_engine


initialize_earth_engine()

def get_soil_type(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])

        soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")

        value = soil.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=250,
            maxPixels=1e9
        )

        soil_class = value.get("b0")

        # Handle null safely
        if soil_class is None:
            return {"soil_type": 0}

        soil_class = soil_class.getInfo()

        # 🔥 Mapping
        if soil_class in [1, 2]:
            soil_type = 1   # Sandy
        elif soil_class in [3, 4, 5]:
            soil_type = 2   # Loamy
        elif soil_class in [6, 7]:
            soil_type = 3   # Clayey
        else:
            soil_type = 0   # Unknown

        return {
            "soil_type": soil_type
        }

    except Exception as e:
        print(f"Soil Type Error: {e}")

        raise