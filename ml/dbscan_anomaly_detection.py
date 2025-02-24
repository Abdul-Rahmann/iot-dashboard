import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import joblib


# ====================
#  DATA PROCESSING
# ====================

def preprocess_data(file_path, scaler_path="data/models/scaler.joblib"):
    """
    Preprocess CSV data for DBSCAN clustering (batch mode).
    Ensures consistent scaling by saving the fitted scaler.

    Args:
        file_path (str): Path to input CSV file.
        scaler_path (str): Path to save fitted scaler.

    Returns:
        normalized_data (np.ndarray): Normalized features.
        data (pd.DataFrame): Original data for mapping.
    """
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values

    # Fit and save the StandardScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(features)
    joblib.dump(scaler, scaler_path)

    print(f"Scaler fitted and saved to {scaler_path}.")
    return normalized_data, data


def preprocess_live_data(data, scaler_path="data/models/scaler.joblib"):
    """
    Preprocess live IoT data using the saved scaler.

    Args:
        data (pd.DataFrame): Dataframe with ['temperature', 'vibration', 'pressure'].
        scaler_path (str): Path to load the trained scaler.

    Returns:
        normalized_data (np.ndarray): Normalized real-time features.
    """
    features = data[['temperature', 'vibration', 'pressure']].values

    # Load the fitted scaler and transform live data
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise ValueError(f"Scaler file not found at {scaler_path}. Train the model first.")

    return scaler.transform(features)


# ====================
#  MODEL TRAINING & TUNING
# ====================

def tune_dbscan_parameters(data, eps_range, min_samples_range, verbose=True):
    """
    Tune DBSCAN hyperparameters for batch data.
    Args:
        data (np.ndarray): Dataset for DBSCAN clustering.
        eps_range (iterable): Range of epsilon values to search.
        min_samples_range (iterable): Range of min_samples values.
        verbose (bool): Print tuning progress.

    Returns:
        best_model (DBSCAN): Optimized DBSCAN model.
        best_labels (np.ndarray): Cluster labels from the best model.
        best_eps (float): Optimal epsilon value.
        best_min_samples (int): Optimal min_samples value.
        best_score (float): Best silhouette score achieved.
    """
    best_eps = None
    best_min_samples = None
    best_score = -1
    best_model = None
    best_labels = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            # Skip if all points are anomalies or a single cluster
            if len(set(labels)) <= 1:
                continue

            try:
                score = silhouette_score(data, labels)
            except ValueError:
                continue  # Skip invalid configurations

            if verbose:
                print(f"eps={eps:.3f}, min_samples={min_samples}: Silhouette Score={score:.4f}")

            if score > best_score:
                best_eps = eps
                best_min_samples = min_samples
                best_score = score
                best_model = dbscan
                best_labels = labels

    return best_model, best_labels, best_eps, best_min_samples, best_score


# ====================
#  MODEL INFERENCE
# ====================

def predict_clusters_live(model, data):
    """
    Apply a pre-trained DBSCAN model to detect clusters in real-time data.
    Args:
        model (DBSCAN): Pre-trained DBSCAN model.
        data (np.ndarray): Real-time normalized data.

    Returns:
        labels (np.ndarray): Cluster labels predicted by the model.
    """
    if model is None:
        raise ValueError("DBSCAN model is not loaded. Please load the model first.")
    return model.fit_predict(data)


# ====================
#  MODEL UTILITIES
# ====================

def load_dbscan_model(model_path):
    """
    Load a pre-trained DBSCAN model.
    """
    try:
        model = joblib.load(model_path)
        print(f"DBSCAN model loaded successfully from {model_path}.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return None


def save_dbscan_model(model, model_path):
    """
    Save a trained DBSCAN model.
    """
    joblib.dump(model, model_path)
    print(f"DBSCAN model saved successfully to {model_path}.")


# ====================
#  MAIN SCRIPT
# ====================

if __name__ == "__main__":
    file_path = 'data/mock_data.csv'
    model_path = 'data/models/dbscan_model.joblib'

    # Fit and save the scaler
    normalized_data, original_data = preprocess_data(file_path)

    # Tune DBSCAN
    eps_range = np.arange(0.1, 1.0, 0.05)
    min_samples_range = range(3, 10)
    model, labels, best_eps, best_min_samples, best_score = tune_dbscan_parameters(
        normalized_data, eps_range, min_samples_range
    )

    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}")
    print(f"Best Silhouette Score: {best_score}, Total Clusters Formed: {len(set(labels))}")

    # Save results
    output_df = original_data.copy()
    output_df['Cluster_Label'] = labels
    output_df.to_csv('data/detected_clusters.csv', index=False)

    # Save model and scaler
    save_dbscan_model(model, model_path)