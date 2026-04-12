import ee

from satellite_features.gee_client import initialize_earth_engine


initialize_earth_engine()

def get_sentinel1_features(lat, lon, start_date="2025-05-01", end_date="2025-05-31"):
    try:
        point = ee.Geometry.Point([lon, lat])

        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .median()
        )

        vv = collection.select('VV')
        vh = collection.select('VH')

        ratio = vv.divide(vh).rename("VV_VH_ratio")

        image = collection.addBands(ratio)

        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        )

        data = values.getInfo()

        features = {
            "VV_mean": data.get("VV", 0),
            "VH_mean": data.get("VH", 0),
            "VV_VH_ratio_mean": data.get("VV_VH_ratio", 0),
        }

        return features

    except Exception as e:
        print(f"Sentinel-1 Error: {e}")

        raise