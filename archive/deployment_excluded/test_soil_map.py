import rasterio
from pyproj import Transformer

# Open raster
src = rasterio.open("tileSG-001-050_1-3.tif")

print("CRS:", src.crs)
print("Bounds:", src.bounds)

# Your lat/lon (Pune)
lat = 18.5204
lon = 73.8567

# 🔥 Convert lat/lon → raster CRS
transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
x, y = transformer.transform(lon, lat)

print("Converted coords:", x, y)

# Now get value
try:
    row, col = src.index(x, y)
    value = src.read(1)[row, col]
    print(f"Clay value at ({lat}, {lon}):", value)
except Exception as e:
    print("Error:", e)