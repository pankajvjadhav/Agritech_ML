import ee

from satellite_features.gee_client import initialize_earth_engine


initialize_earth_engine()

def get_sentinel2_features(lat, lon, start_date="2025-05-01", end_date="2025-05-31"):
    try:
        point = ee.Geometry.Point([lon, lat])

        collection = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
        )

        blue = collection.select("B2")
        red = collection.select("B4")
        nir = collection.select("B8")
        swir1 = collection.select("B11")
        swir2 = collection.select("B12")

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")

        bsi = (
            (swir1.add(red).subtract(nir.add(blue)))
            .divide(swir1.add(red).add(nir).add(blue))
            .rename("BSI")
        )

        nbr2 = swir1.subtract(swir2).divide(swir1.add(swir2)).rename("NBR2")

        image = collection.addBands([ndvi, ndmi, bsi, nbr2])

        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        )

        data = values.getInfo()

        features = {
            "NDVI_mean": data.get("NDVI", 0),
            "NDMI_mean": data.get("NDMI", 0),
            "BSI_mean": data.get("BSI", 0),
            "NBR2_mean": data.get("NBR2", 0),
        }

        return features

    except Exception as e:
        print(f"Sentinel-2 Error: {e}")

        raise