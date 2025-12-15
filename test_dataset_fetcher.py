"""
Test script for dataset-based nutrient prediction.
This demonstrates how to use the new /predict/from-dataset endpoint.
"""

from app.data_fetcher import fetch_from_dataset
import pandas as pd

# Test the dataset fetcher function
print("Testing dataset fetcher...")
print("=" * 50)

# Load dataset to see how many rows are available
dataset_path = "static_dataset.csv"
df = pd.read_csv(dataset_path)
print(f"Dataset has {len(df)} rows (indices 0-{len(df)-1})")
print("\nDataset preview:")
print(df.head())
print("\n" + "=" * 50)

# Test fetching data by index
for index in range(min(3, len(df))):
    print(f"\nTesting index {index}:")
    try:
        features, meta = fetch_from_dataset(index)
        print(f"Successfully fetched data from index {index}")
        print(f"Meta: {meta}")
        print(f"Features: {features}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 50)
print("To test the API endpoint, use:")
print("POST /predict/from-dataset")
print('Body: {"index": 0}')
print("\nExample with curl:")
print('curl -X POST "http://localhost:8000/predict/from-dataset" \\')
print('  -H "Content-Type: application/json" \\')
print('  -d \'{"index": 0}\'')


