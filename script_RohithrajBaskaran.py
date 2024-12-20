import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/consumer_data.csv')

# Preprocessing
# Convert numeric columns
numeric_columns = ['order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode categorical variables
categorical_columns = ['department', 'department_id', 'reordered']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Group by user_id to create user-level features
user_features = df.groupby('user_id').agg({
    'order_number': 'max',
    'order_dow': 'mean',
    'order_hour_of_day': 'mean',
    'days_since_prior_order': 'mean',
    'add_to_cart_order': 'mean',
    'reordered': 'mean',
    'department': lambda x: x.value_counts().index[0],  # Most frequent department
    'department_id': lambda x: x.value_counts().index[0]
}).reset_index()

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = user_features.drop('user_id', axis=1)
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# K-means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualization
plt.figure(figsize=(15, 6))

# K-means Clustering Visualization
plt.subplot(121)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.colorbar(scatter1)

# DBSCAN Clustering Visualization
plt.subplot(122)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.colorbar(scatter2)
plt.tight_layout()
plt.show()

# Cluster Analysis Function
def analyze_clusters(labels, method_name):
    print(f"\n{method_name} Clustering Analysis:")
    cluster_count = len(np.unique(labels))
    print("Number of clusters:", cluster_count)
    
    df_clustered = user_features.copy()
    df_clustered['cluster'] = labels
    
    for cluster in range(cluster_count):
        if cluster == -1 and method_name == "DBSCAN":
            print("\nNoise Points")
        else:
            print(f"\nCluster {cluster} Characteristics:")
        
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        print("Cluster Size:", len(cluster_data))
        
        # Cluster-specific statistics
        print("Average Order Number:", cluster_data['order_number'].mean())
        print("Average Day of Week:", cluster_data['order_dow'].mean())
        print("Average Order Hour:", cluster_data['order_hour_of_day'].mean())
        print("Average Days Since Prior Order:", cluster_data['days_since_prior_order'].mean())
        print("Average Items in Cart:", cluster_data['add_to_cart_order'].mean())
        print("Reorder Rate:", cluster_data['reordered'].mean())
        
        # Distribution of categorical features
        print("\nDominant Department:")
        print(cluster_data['department'].value_counts(normalize=True).head(1))

# Perform cluster analysis
analyze_clusters(kmeans_labels, "K-Means")
analyze_clusters(dbscan_labels, "DBSCAN")

# Silhouette Score to measure how good the shape of our clusters are
# kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
# print(f"\nK-Means Silhouette Score: {kmeans_silhouette:.4f}")