import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Initialize a scaler for reuse
scaler = MinMaxScaler()


# ====================
#  DATA PROCESSING
# ====================
def preprocess_data(file_path):
    """
    Preprocess CSV data for DBSCAN clustering (batch mode).
    Args:
        file_path (str): Path to input CSV file.

    Returns:
        normalized_data (np.ndarray): Normalized features.
        data (pd.DataFrame): Original data for mapping.
    """
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values
    normalized_data = scaler.fit_transform(features)
    return normalized_data, data


def preprocess_live_data(data, scaler=scaler):
    """
    Preprocess live IoT data for clustering (real-time mode).
    Args:
        data (pd.DataFrame): Dataframe with embedded metrics ['temperature', 'vibration', 'pressure'].
        scaler (MinMaxScaler): Fitted MinMaxScaler for normalization.

    Returns:
        normalized_data (np.ndarray): Normalized real-time features.
    """
    features = data[['temperature', 'vibration', 'pressure']].values
    return scaler.fit_transform(features)


# ====================
#  MODEL TRAINING & TUNING
# ====================
def tune_dbscan_parameters(data, eps_range, min_samples_range, verbose=True):
    """
    Tune DBSCAN hyperparameters for batch data.
    Args:
        data (np.ndarray): Dataset for DBSCAN clustering.
        eps_range (iterable): Range of epsilon values to search.
        min_samples_range (iterable): Range of minimum samples to search.
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

            # Skip if all points are classified as anomalies or a single cluster
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
        data (np.ndarray): Real-time normalized data for clustering.

    Returns:
        labels (np.ndarray): Cluster labels predicted by the model.
    """
    if model is None:
        raise ValueError("DBSCAN model is not loaded. Please load the model first.")
    return model.fit_predict(data)  # Assign cluster labels for real-time data


# ====================
#  MODEL UTILITIES
# ====================
def load_dbscan_model(model_path):
    """
    Load a pre-trained DBSCAN model from disk.
    Args:
        model_path (str): Path to the trained model file.

    Returns:
        model (DBSCAN): Loaded DBSCAN model.
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
    Save a trained DBSCAN model to disk.
    Args:
        model (DBSCAN): Trained model to save.
        model_path (str): Path to save the model file.
    """
    joblib.dump(model, model_path)
    print(f"DBSCAN model saved successfully to {model_path}.")


# ====================
#  VISUALIZATION
# ====================
def visualize_clusters_2D(data, labels):
    """
    Visualize clusters in 2D using PCA dimensionality reduction.
    Args:
        data (np.ndarray): Input feature data for visualization.
        labels (np.ndarray): Cluster labels for the data.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            color = 'red'
            label_name = 'Anomaly'
        else:
            color = plt.cm.jet(float(label) / len(unique_labels))
            label_name = f'Cluster {label}'
        plt.scatter(
            reduced_data[labels == label, 0],
            reduced_data[labels == label, 1],
            c=color,
            label=label_name,
            s=40, alpha=0.8
        )

    plt.title("DBSCAN Clustering (2D Reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_clusters_interactive_3D(data, labels):
    """
    Visualize clusters in 3D using PCA dimensionality reduction.
    Args:
        data (np.ndarray): Input feature data for visualization.
        labels (np.ndarray): Cluster labels for the data.
    """
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)
    cluster_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2', 'PCA3'])
    cluster_df['Cluster'] = labels.astype(str)
    fig = px.scatter_3d(
        cluster_df,
        x='PCA1', y='PCA2', z='PCA3',
        color='Cluster',
        title="Interactive 3D Clustering Visualization",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.show()


# ====================
#  MAIN SCRIPT
# ====================
if __name__ == "__main__":
    file_path = 'data/mock_data.csv'
    normalized_data, original_data = preprocess_data(file_path)

    # Hyperparameter tuning
    eps_range = np.arange(0.1, 0.5, 0.05)
    min_samples_range = range(3, 10)
    model, labels, best_eps, best_min_samples, best_score = tune_dbscan_parameters(
        normalized_data, eps_range, min_samples_range
    )
    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}")
    print(f"Best Silhouette Score: {best_score}")

    # Save the results
    output_df = original_data.copy()
    output_df['Cluster_Label'] = labels
    output_df.to_csv('data/detected_clusters.csv', index=False)

    # Visualize and save the model
    visualize_clusters_2D(normalized_data, labels)
    save_dbscan_model(model, 'data/models/dbscan_model.joblib')
