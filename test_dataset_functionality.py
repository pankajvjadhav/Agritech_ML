"""
Test script to verify dataset functionality is working
"""
import sys
from app.data_fetcher import fetch_from_dataset, fetch_real_time_data
from app.predictor import make_prediction
from app.model_loader import model

print("=" * 60)
print("Testing Dataset Functionality")
print("=" * 60)

# Test 1: Fetch by specific index
print("\n1. Testing fetch_from_dataset with index 0:")
try:
    features, meta = fetch_from_dataset(0)
    print(f"   [OK] Success! Loaded row {meta['row_index']}")
    print(f"   [OK] Total rows in dataset: {meta['total_rows']}")
    print(f"   [OK] Features: {len(features)} features loaded")
    print(f"   [OK] Sample values: pH={features['pH_0_30']}, NDVI={features['ndvi_mean_90d']}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

# Test 2: Fetch random row with lat/lon
print("\n2. Testing fetch_real_time_data with lat/lon (random selection):")
try:
    features, meta = fetch_real_time_data(28.6, 77.2, 1.0, None, use_dataset=True)
    print(f"   [OK] Success! Randomly selected row {meta['row_index']}")
    print(f"   [OK] Source: {meta['source']}")
    print(f"   [OK] Location provided: lat={meta['location']['lat']}, lon={meta['location']['lon']}")
    print(f"   [OK] Features: {len(features)} features loaded")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

# Test 3: Make prediction with dataset data
print("\n3. Testing prediction with dataset data:")
try:
    if model is None:
        print("   ⚠ Model not loaded, skipping prediction test")
    else:
        result = make_prediction(model, features)
        print(f"   [OK] Prediction successful!")
        print(f"   [OK] Nutrients predicted: {len(result)} nutrients")
        print(f"   [OK] Sample predictions:")
        for nutrient in ['N', 'P', 'K']:
            if nutrient in result:
                val = result[nutrient].get('value', 'N/A')
                print(f"      {nutrient}: {val} {result[nutrient].get('unit', '')}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test multiple random selections
print("\n4. Testing multiple random selections:")
try:
    selected_indices = []
    for i in range(3):
        features, meta = fetch_real_time_data(28.6, 77.2, 1.0, None, use_dataset=True)
        selected_indices.append(meta['row_index'])
    print(f"   [OK] Selected {len(selected_indices)} random rows: {selected_indices}")
    print(f"   [OK] (Note: May have duplicates due to random selection)")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! [OK]")
print("=" * 60)

