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
        scaler (StandardScaler): Scaler for inverse transformation if needed.
    """
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features)
    return normalized_data, scaler, data


def fit_kmeans(data, n_clusters=3):
    """
    Fits a K-Means model to cluster the data.
    Args:
        data (numpy.ndarray): Normalized data to cluster.
        n_clusters (int): Number of clusters.
    Returns:
        kmeans (KMeans): Fitted K-Means model.
        distances (numpy.ndarray): Distance of each point to its nearest cluster center.
        silhouette (float): Silhouette Score for the clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    distances = np.min(kmeans.transform(data), axis=1)
    silhouette = silhouette_score(data, kmeans.labels_)
    return kmeans, distances, silhouette


def detect_anomalies(distances, std_multiplier=2.0):
    """
    Detect anomalies based on a dynamic distance threshold.
    Args:
        distances (numpy.ndarray): Distances of each point to its cluster center.
        std_multiplier (float): Multiplier for standard deviation.
    Returns:
        anomalies (numpy.ndarray): Indices of detected anomalies.
        threshold (float): Calculated threshold for classification.
    """
    threshold = np.mean(distances) + std_multiplier * np.std(distances)
    anomalies = np.where(distances > threshold)[0]
    return anomalies, threshold


if __name__ == "__main__":
    # Settings
    file_path = 'data/mock_data.csv'
    n_clusters = 6
    std_multiplier = 2.0

    # Preprocessing
    data, scaler, original_data = preprocess_data(file_path)

    # K-Means Clustering
    kmeans, distances, silhouette = fit_kmeans(data, n_clusters=n_clusters)

    # Anomaly Detection
    anomalies, threshold = detect_anomalies(distances, std_multiplier=std_multiplier)

    # Visualization
    visualize_clusters(data, kmeans, anomalies)

    # Output Summary
    num_points = len(data)
    num_anomalies = len(anomalies)
    print(f"Number of data points: {num_points}")
    print(f"Number of anomalies detected: {num_anomalies}")
    print(f"Indices of anomalies: {anomalies}")
    print(f"Threshold for anomaly detection: {threshold:.3f}")
    print(f"Silhouette Score: {silhouette:.4f}")

    cluster_counts = pd.Series(kmeans.labels_).value_counts()
    print("\nCluster Sizes:")
    print(cluster_counts)

    output_df = original_data.copy()
    output_df['Cluster_Label'] = kmeans.labels_
    output_df['Distance_to_Center'] = distances
    output_df['Is_Anomaly'] = False
    output_df.loc[anomalies, 'Is_Anomaly'] = True
    output_df.to_csv('data/anomaly_detection.csv', index=False)
    print("\nResults saved to 'data/anomaly_detection.csv'.")
