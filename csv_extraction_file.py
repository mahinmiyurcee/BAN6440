# extract_10000_rows.py

"""
Script to extract the first 10,000 valid rows from the NYC Taxi Parquet dataset.
The output is saved as a CSV file for easier downstream use in clustering applications.
"""

import pandas as pd
import os

# File paths
input_parquet = "C:/Users/Dell/Downloads/yellow_tripdata_2025-05.parquet"
output_csv = "C:.nyc_taxi_10000.csv"

def extract_sample_data(parquet_path, output_path, num_rows=10000):
    try:
        print("Reading Parquet file...")
        df = pd.read_parquet(parquet_path)

        # Select relevant features for clustering
        features = ['trip_distance', 'fare_amount', 'passenger_count', 'total_amount']

        print("Filtering and extracting first {} valid rows...".format(num_rows))
        df_subset = df[features].dropna().head(num_rows)

        print("Saving to CSV...")
        df_subset.to_csv(output_path, index=False)

        print(f"Extraction complete! Saved to: {output_path}")
        return True
    except Exception as e:
        print("Error during extraction:", e)
        return False

if __name__ == "__main__":
    success = extract_sample_data(input_parquet, output_csv)
    if success:
        print("✅ Data extraction script finished successfully.")
    else:
        print("❌ Script failed.")
