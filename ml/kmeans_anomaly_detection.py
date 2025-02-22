from dis import disco

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def visualize_clusters(data, kmeans, anomalies):
    """
       Visualizes clusters and highlights anomalies.
       Args:
           data (numpy.ndarray): The normalized data.
           kmeans (KMeans): Trained K-Means model.
           anomalies (numpy.ndarray): Indices of detected anomalies.
       """
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap="viridis", s=50, alpha=0.5, label="Normal Points")
    plt.scatter(data[anomalies][:, 0], data[anomalies][:, 1], color="red", s=70, label="Anomalies")

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", s=200, marker="X",
                label="Cluster Centers")

    plt.title("Clusters and Anomalies")
    plt.legend()
    plt.show()
def preprocess_data(file_path):
    """
       Preprocesses the IoT data by normalizing sensor values.
       Args:
           file_path (str): Path to the IoT data file (CSV).
       Returns:
           normalized_data (numpy.ndarray): Normalized sensor data.
           scaler (MinMaxScaler): Scaler for inverse transformation if needed.
    """
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features)

    return normalized_data, scaler

def fit_kmeans(data, n_clusters=3):
    """
       Fits a K-Means model to cluster the data.
       Args:
           data (numpy.ndarray): Normalized data to cluster.
           n_clusters (int): Number of clusters.
       Returns:
           kmeans (KMeans): Fitted K-Means model.
           distances (numpy.ndarray): Distance of each point to its nearest cluster center.
       """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    distances = np.min(kmeans.transform(data), axis=1)

    return kmeans, distances

def detect_anomaly(distances, threshold):
    """
       Detect anomalies based on a given distance threshold.
       Args:
           distances (numpy.ndarray): Distances of each point to its cluster center.
           threshold (float): Distance threshold for anomaly detection.
       Returns:
           anomalies (numpy.ndarray): Indices of the detected anomalies.
       """
    anomalies = np.where(distances > threshold)[0]
    return anomalies

if __name__ == "__main__":
    file_path = 'data/mock_data.csv'
    data, scaler = preprocess_data(file_path)

    n_clusters = 6
    kmeans, distances = fit_kmeans(data, n_clusters=n_clusters)

    threshold = 0.60
    anomalies = detect_anomaly(distances, threshold)
    visualize_clusters(data, kmeans, anomalies)

    print(f"Number of data points: {len(data)}")
    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Indices of anomalies: {anomalies}")

    pd.DataFrame({"distance": distances, "is_anomaly": distances > threshold}).to_csv('data/anomaly_detection.csv', index=False)
