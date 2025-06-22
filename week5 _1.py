import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules



# Read the CSV file
retail_df = pd.read_csv("retail_data.csv")

# Create RFM features per customer
snapshot_date = pd.to_datetime(retail_df['InvoiceDate']).max() + timedelta(days=1)
retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'])

rfm = retail_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}).reset_index()

# EDA
sns.pairplot(rfm[['Recency', 'Frequency', 'Monetary']])
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm')
plt.title("RFM Correlation Heatmap")
plt.show()

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Elbow Method for K selection
wcss = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
plt.plot(range(2, 10), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Silhouette Score
score = silhouette_score(X_scaled, rfm['Cluster'])
print(f"Silhouette Score: {score:.2f}")

# Cluster summary
print(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
rfm['PCA1'] = pca_components[:,0]
rfm['PCA2'] = pca_components[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=rfm, palette='Set2')
plt.title("Customer Segments (PCA Projection)")
plt.show()

# 1Market Basket Analysis - Recommendation System
# Prepare basket data
grouped = retail_df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0)
basket = grouped.set_index('InvoiceNo')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

