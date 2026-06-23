# prompt: Create synthetic data for clustering in ESG Context with at least reasonable 5 dimensions. Then show the data. Then do clustering - K-Means and Hierarchical clustering. Visualize the output clusters from both methods. Report and discuss the results. When doing K-Mean, find the optimal K number using silouhete score and elbow methods and show its chart as well.

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create Synthetic Data for ESG Clustering
# We'll create data points with features that could represent ESG aspects.
# Features: Environmental Impact, Social Impact, Governance Score, Employee Wellbeing, Sustainable Practices

n_samples = 300
n_features = 5
n_centers  = 4  # Assuming 4 underlying 'types' of companies based on ESG

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers,
                  cluster_std=1.7, random_state=11)

# Let's scale the data, which is often important for clustering algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame for easier handling and visualization
feature_names = ['Environmental_Impact', 'Social_Impact', 'Governance_Score', 'Employee_Wellbeing', 'Sustainable_Practices']
df = pd.DataFrame(X_scaled, columns=feature_names)

print("Synthetic ESG Data (Scaled):")
print(df.head())
print("\nData Description:")
print(df.describe())

# 2. K-Means Clustering
# Find optimal K using Elbow and Silhouette methods

# Elbow Method
sse = []
k_range = range(1, 11) # Test K from 1 to 10
for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  kmeans.fit(X_scaled)
  sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Silhouette Method
silhouette_scores = []
k_range_silhouette = range(2, 11) # Silhouette score is not defined for K=1
for k in k_range_silhouette:
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  kmeans.fit(X_scaled)
  score = silhouette_score(X_scaled, kmeans.labels_)
  silhouette_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(k_range_silhouette, silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range_silhouette)
plt.grid(True)
plt.show()

# Based on the plots (visually inspecting the 'elbow' and the peak silhouette score),
# let's choose an optimal K. Let's assume 3 looks reasonable for demonstration.
optimal_k = 4 # Adjust based on your visual inspection

kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_optimal.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# 3. Hierarchical Clustering
# We'll use Agglomerative Clustering

hierarchical = AgglomerativeClustering(n_clusters=optimal_k) # Using the same number of clusters as K-Means for comparison
hierarchical_labels = hierarchical.fit_predict(X_scaled)
df['Hierarchical_Cluster'] = hierarchical_labels

# 4. Visualize Clusters

# Since we have 5 dimensions, we can't visualize all at once directly.
# We'll use a pair plot for pairwise visualizations, colored by cluster.
# Due to the number of dimensions, the pair plot might be slow or dense.
# Alternatively, we can use dimensionality reduction (like PCA) for 2D/3D visualization.
# Let's start with a pair plot of a subset of features or the first few principal components.

# For simplicity, let's visualize using the first 2 principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='viridis', legend='full')
plt.title(f'K-Means Clustering (K={optimal_k}) on PCA Components')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Hierarchical_Cluster', palette='viridis', legend='full')
plt.title(f'Hierarchical Clustering (K={optimal_k}) on PCA Components')

plt.tight_layout()
plt.show()

# Now we could also visualize using a subset of original features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Environmental_Impact', y='Social_Impact', hue='KMeans_Cluster', palette='viridis', legend='full')
plt.title(f'K-Means Clustering (K={optimal_k})')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Environmental_Impact', y='Social_Impact', hue='Hierarchical_Cluster', palette='viridis', legend='full')
plt.title(f'Hierarchical Clustering (K={optimal_k})')

plt.tight_layout()
plt.show()

# 5. Report and Discuss Results

print("\n--- Clustering Results ---")

print("\nK-Means Cluster Sizes:")
print(df['KMeans_Cluster'].value_counts().sort_index())

print("\nHierarchical Cluster Sizes:")
print(df['Hierarchical_Cluster'].value_counts().sort_index())

print("\nMean values per K-Means Cluster:")
print(df.groupby('KMeans_Cluster')[feature_names].mean())

print("\nMean values per Hierarchical Cluster:")
print(df.groupby('Hierarchical_Cluster')[feature_names].mean())

print(f"\nThe visualizations on the first two principal components show how the clusters separate in a lower-dimensional space.")
print("K-Means aims to partition the data into K distinct, non-overlapping subgroups, minimizing variance within each cluster.")
print("Hierarchical clustering builds a hierarchy of clusters. Agglomerative (bottom-up) starts with each point as a cluster and merges them.")
print("\nLooking at the mean values per cluster:")
print("- We can observe the characteristics of each cluster based on the average values of the ESG features.")
print(f"- For example, a cluster might have high 'Environmental_Impact' and low 'Governance_Score', representing a certain type of company.")
print("- Comparing the mean values between K-Means and Hierarchical clusters can reveal similarities and differences in how the algorithms grouped the data.")
print("- K-Means typically produces more spherical clusters and requires pre-specifying K.")
print("- Hierarchical clustering does not strictly require pre-specifying K (though we did for visualization comparison) and can reveal hierarchical relationships.")

print("\nFurther steps could involve:")
print("- Trying different clustering algorithms (e.g., DBSCAN).")
print("- Evaluating clustering performance using internal or external metrics (if ground truth labels were available).")
print("- Interpreting clusters in the context of real-world business or investment decisions.")
