from app.data_fetcher import fetch_real_time_data

# Test with dummy data (no credentials)
lat = 28.6139
lon = 77.2090
area_ha = 1.2

print("Testing with dummy satellite data:")
data = fetch_real_time_data(lat, lon, area_ha)
print(data)

# To test with real data, 
satellite_api_key = "753fcd334b364c2041ad8d63ef8ff608"
print("Testing with real satellite data:")
try:
    data_real = fetch_real_time_data(lat, lon, area_ha, satellite_api_key)
    print(data_real)
except Exception as e:
    print(f"Error fetching real data: {e}")