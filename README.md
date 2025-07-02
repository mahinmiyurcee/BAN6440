# K-Means Clustering on NYC Taxi Trip Data

This project is part of the **BAN 6440: Machine Learning for Analytics** coursework. It demonstrates the implementation of a K-Means clustering application in Python using a subset of the NYC Yellow Taxi Trip dataset.

## Project Structure

- `csv_extraction_file.py` - Python script to extract subset of 10,000 rows from the parquet file of May 2025 NYC taxi trips.
- `nyc_taxi_10000.csv` – Sample dataset containing the first 10,000 rows of May 2025 NYC taxi trips.
- `kmeans_clustering.py` – Main Python script to preprocess the dataset, perform clustering, and visualize the results.
- `test_kmeans_clustering.py` – Unit tests to verify functionality and robustness of the clustering pipeline.

## Summary
The goal of this project was to use the K-Means algorithm to cluster taxi trip data based on quantitative trip attributes. Principal Component Analysis (PCA) was used to visualize the resulting clusters in two dimensions. 

## Features Used for Clustering
- Trip distance
- Fare amount
- Passenger count
- Total amount

## Tools & Libraries
- Python (3.12)
- scikit-learn
- pandas
- seaborn
- matplotlib
- PyCharm IDE

## Unit Testing
Unit tests were written using Python's built-in `unittest` framework to ensure the reliability of core functionalities such as data loading, clustering execution, and shape validation.

## Note
The dataset was originally provided in Parquet format. A subset of 10,000 rows was extracted and saved in CSV format to improve performance and simplify processing.
