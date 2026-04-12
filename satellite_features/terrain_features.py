import ee

from satellite_features.gee_client import initialize_earth_engine


initialize_earth_engine()

def get_terrain_features(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])

        dem = ee.Image("USGS/SRTMGL1_003")

        elevation = dem.select("elevation")
        slope = ee.Terrain.slope(elevation)

        image = elevation.addBands(slope)

        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        )

        data = values.getInfo()

        features = {
            "elevation": data.get("elevation", 0),
            "slope": data.get("slope", 0),
        }

        return features

    except Exception as e:
        print(f"Terrain Error: {e}")

        raise