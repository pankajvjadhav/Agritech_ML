"""
Test the API endpoints to verify they work with the dataset
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing API Endpoints")
print("=" * 60)

# Test 1: Check if server is running
print("\n1. Checking if server is running...")
try:
    response = requests.get(f"{BASE_URL}/", timeout=5)
    if response.status_code == 200:
        print(f"   [OK] Server is running!")
        print(f"   [OK] Response: {response.json()}")
    else:
        print(f"   [FAIL] Server returned status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   [FAIL] Server not running: {e}")
    print("   Please start the server with: python main.py")
    exit(1)

# Test 2: Test /predict/from-dataset endpoint
print("\n2. Testing /predict/from-dataset endpoint (index 0):")
try:
    payload = {"index": 0}
    response = requests.post(f"{BASE_URL}/predict/from-dataset", json=payload, timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Request successful!")
        print(f"   [OK] Success: {data.get('success')}")
        print(f"   [OK] Meta - Row index: {data.get('meta', {}).get('row_index')}")
        print(f"   [OK] Meta - Source: {data.get('meta', {}).get('source')}")
        if 'predictions' in data:
            preds = data['predictions']
            print(f"   [OK] Predictions: {len(preds)} nutrients")
            for nutrient in ['N', 'P', 'K']:
                if nutrient in preds:
                    val = preds[nutrient].get('value', 'N/A')
                    print(f"      {nutrient}: {val} {preds[nutrient].get('unit', '')}")
    else:
        print(f"   [FAIL] Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 3: Test /predict/realtime endpoint (should use random dataset row)
print("\n3. Testing /predict/realtime endpoint (random dataset selection):")
try:
    payload = {
        "lat": 28.6139,
        "lon": 77.2090,
        "area_ha": 1.2
    }
    response = requests.post(f"{BASE_URL}/predict/realtime", json=payload, timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Request successful!")
        print(f"   [OK] Success: {data.get('success')}")
        meta = data.get('meta', {})
        print(f"   [OK] Meta - Source: {meta.get('source')}")
        print(f"   [OK] Meta - Row index: {meta.get('row_index')}")
        print(f"   [OK] Meta - Total rows: {meta.get('total_rows')}")
        if 'predictions' in data:
            preds = data['predictions']
            print(f"   [OK] Predictions: {len(preds)} nutrients")
            for nutrient in ['N', 'P', 'K']:
                if nutrient in preds:
                    val = preds[nutrient].get('value', 'N/A')
                    print(f"      {nutrient}: {val} {preds[nutrient].get('unit', '')}")
    else:
        print(f"   [FAIL] Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 4: Test /predict/dataset-info endpoint
print("\n4. Testing /predict/dataset-info endpoint:")
try:
    response = requests.get(f"{BASE_URL}/predict/dataset-info", timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Request successful!")
        print(f"   [OK] Total rows: {data.get('total_rows')}")
        print(f"   [OK] Available indices: {data.get('available_indices')}")
        print(f"   [OK] Columns: {len(data.get('columns', []))} columns")
    else:
        print(f"   [FAIL] Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 5: Test multiple realtime calls (should get different random rows)
print("\n5. Testing multiple /predict/realtime calls (checking randomness):")
try:
    selected_indices = []
    for i in range(3):
        payload = {
            "lat": 28.6139 + i * 0.1,  # Slightly different locations
            "lon": 77.2090 + i * 0.1,
            "area_ha": 1.2
        }
        response = requests.post(f"{BASE_URL}/predict/realtime", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            row_idx = data.get('meta', {}).get('row_index')
            selected_indices.append(row_idx)
        time.sleep(0.5)  # Small delay between requests
    
    print(f"   [OK] Selected row indices: {selected_indices}")
    print(f"   [OK] (Note: Random selection may have duplicates)")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n" + "=" * 60)
print("API Endpoint Tests Complete!")
print("=" * 60)


