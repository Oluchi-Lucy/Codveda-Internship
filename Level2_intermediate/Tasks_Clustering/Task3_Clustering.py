import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("1) iris.csv")

# Display the first five rows
print(df.head())

# Display the column names
print(df.columns)

# Display info about the dataset
print(df.info())
from sklearn.preprocessing import StandardScaler

# Select only the numeric columns (drop 'species' since it's text, not a measurement)
features = df.drop('species', axis=1)

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print(scaled_features[:5])  # show first 5 rows after scaling
from sklearn.cluster import KMeans

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.savefig("elbow_method.png")
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='saddlebrown')
plt.title("Elbow Method for Optimal K", fontweight='bold')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.savefig("elbow_method.png")
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

df['cluster'] = clusters

plt.figure(figsize=(8, 6))
plt.scatter(df['petal_length'], df['petal_width'], c=df['cluster'], cmap='viridis')
plt.title("K-Means Clustering of Iris Flowers", fontweight='bold')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.savefig("kmeans_clusters.png")
plt.show()