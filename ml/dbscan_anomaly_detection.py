import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    """
       Preprocess IoT data into normalized feature space for DBSCAN.
       Args:
           file_path (str): Path to the CSV file containing IoT data.
       Returns:
           normalized_data (np.ndarray): Normalized sensor data.
           scaler (MinMaxScaler): Scaler used for normalization.
       """
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(features)
    return normalized_data, scaler

def detect_anomaly_dbscan(data, eps=0.2, min_samples=5):
    """
       Runs DBSCAN for anomaly detection on IoT data.
       Args:
           data (np.ndarray): Preprocessed (normalized) data.
           eps (float): Maximum distance between points in a cluster.
           min_samples (int): Minimum data points required to form a cluster.
       Returns:
           labels (np.ndarray): Cluster assignments for each point (-1 indicates anomalies).
       """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

if __name__ == "__main__":
    file_path = 'data/mock_data.csv'
    normalized_data, scaler = preprocess_data(file_path)

    eps_value = 0.15
    min_samples_value = 5
    labels = detect_anomaly_dbscan(normalized_data, eps=eps_value, min_samples=min_samples_value)

    anomaly_indices = np.where(labels == -1)[0]
    print(f"Total data points: {len(normalized_data)}")
    print(f"Number of anomalies detected: {len(anomaly_indices)}")
    print(f"Indices of anomalies: {anomaly_indices}")

    pd.DataFrame({"labels": labels}).to_csv('data/anomaly_detection_dbscan.csv', index=False)

