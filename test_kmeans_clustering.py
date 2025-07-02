# test_kmeans_clustering.py

import unittest
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TestKMeansClustering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the 10,000-row dataset
        cls.df = pd.read_csv(r"C:\Users\Dell\PycharmProjects\BAN6440_Module_4\.nyc_taxi_10000.csv")
        cls.features = ['trip_distance', 'fare_amount', 'passenger_count', 'total_amount']
        cls.data = cls.df[cls.features].dropna()

        # Standardize data
        cls.scaler = StandardScaler()
        cls.scaled = cls.scaler.fit_transform(cls.data)

    def test_shape_after_scaling(self):
        # Test if the shape remains consistent
        self.assertEqual(self.scaled.shape[0], self.data.shape[0])
        self.assertEqual(self.scaled.shape[1], len(self.features))

    def test_kmeans_clusters(self):
        # Test if KMeans produces the right number of clusters
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(self.scaled)
        self.assertEqual(len(set(labels)), 3)

    def test_pca_dimensions(self):
        # Test PCA dimensionality reduction to 2 components
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.scaled)
        self.assertEqual(reduced.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
