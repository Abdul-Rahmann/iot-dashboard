import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px


def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(features)
    return normalized_data, data


def tune_dbscan_parameters(data, eps_range, min_samples_range, verbose=True):
    best_eps = None
    best_min_samples = None
    best_score = -1
    best_model = None
    best_labels = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            if len(set(labels)) <= 1:
                continue

            try:
                score = silhouette_score(data, labels)
            except ValueError:
                continue

            if verbose:
                print(f"eps={eps:.3f}, min_samples={min_samples}: Silhouette Score={score:.4f}")

            if score > best_score:
                best_eps = eps
                best_min_samples = min_samples
                best_score = score
                best_model = dbscan
                best_labels = labels

    return best_model, best_labels, best_eps, best_min_samples, best_score


def visualize_clusters_2D(data, labels):
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

    plt.title("DBSCAN Clustering (PCA Reduced to 2D)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_clusters_interactive_3D(data, labels):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)
    cluster_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2', 'PCA3'])
    cluster_df['Cluster'] = labels.astype(str)
    fig = px.scatter_3d(
        cluster_df,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color='Cluster',
        title="Interactive 3D Clustering Visualization",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.show()


if __name__ == "__main__":
    file_path = 'data/mock_data.csv'
    normalized_data, original_data = preprocess_data(file_path)
    eps_range = np.arange(0.1, 0.5, 0.05)
    min_samples_range = range(3, 10)

    dbscan_model, labels, best_eps, best_min_samples, best_score = tune_dbscan_parameters(
        normalized_data,
        eps_range,
        min_samples_range,
        verbose=True
    )

    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}")
    print(f"Best Silhouette Score: {best_score}")
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters formed (excluding anomalies): {num_clusters}")
    num_anomalies = list(labels).count(-1)
    print(f"Number of anomalies detected: {num_anomalies}")

    output_df = original_data.copy()
    output_df['Cluster_Label'] = labels
    output_df.to_csv('data/detected_clusters.csv', index=False)
    visualize_clusters_2D(normalized_data, labels)
    visualize_clusters_interactive_3D(normalized_data, labels)
