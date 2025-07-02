# kmeans_clustering.py

"""
K-Means Clustering on NYC Taxi Trip Subset (10,000 rows)
Performs preprocessing, clustering, PCA-based visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load subset dataset

df = pd.read_csv(r"C:\Users\Dell\PycharmProjects\BAN6440_Module_4\.nyc_taxi_10000.csv")
# Print first few rows for reference
print("\n First 5 rows of dataset:")
print(df.head())

# Step 2: Select numerical features for clustering
features = ['trip_distance', 'fare_amount', 'passenger_count', 'total_amount']
data = df[features].dropna()

# Step 3: Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
data['cluster'] = clusters

# Step 5: Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
data['pca1'] = pca_result[:, 0]
data['pca2'] = pca_result[:, 1]

# Step 6: Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='pca1', y='pca2', hue='cluster', palette='viridis')
plt.title("K-Means Clustering on NYC Taxi Trips (10,000 Samples)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
