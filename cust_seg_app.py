import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the dataset (Replace with actual path)
st.title("Customer Segmentation Clustering")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    
    #st.write("Available columns:", list(df.columns))
    selected_features = st.multiselect("Select columns for clustering", df.columns)
    
    if selected_features:
        features = df[selected_features]

        # Determine the optimal number of clusters using Silhouette Score
        best_k = 2
        best_score = -1
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            if score > best_score:
                best_k = k
                best_score = score

        st.write(f'Optimal number of clusters: {best_k} (Silhouette Score: {best_score:.3f})')

        # Apply K-Means Clustering with optimal K
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=100)
        df['Cluster_KMeans'] = kmeans.fit_predict(features)

        # Calculate evaluation metrics
        sil_score_kmeans = silhouette_score(features, df['Cluster_KMeans'])
        db_index_kmeans = davies_bouldin_score(features, df['Cluster_KMeans'])
        st.write(f'K-Means - Silhouette Score: {sil_score_kmeans:.3f}, Davies-Bouldin Index: {db_index_kmeans:.3f}')

        # Visualize the Clusters (K-Means)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], hue=df['Cluster_KMeans'], palette='viridis')
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1] if len(selected_features) > 1 else 'Feature 2')
        plt.title('Customer Segmentation using K-Means')
        st.pyplot(plt)
    else:
        st.write("Please select at least one feature for clustering.")