import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def preprocess_data(file_path):
    """
    Preprocesses data by scaling features to zero mean and unit variance.
    Args:
        file_path (str): Path to the data file (CSV).
    Returns:
        normalized_data (numpy.ndarray): Preprocessed (scaled) data as NumPy array.
        scaler (StandardScaler): Fitted scaler object (for reconstruction).
        original_data (pandas.DataFrame): Raw dataset.
    """
    data = pd.read_csv(file_path)
    features = data[['temperature', 'vibration', 'pressure']].values
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features)
    return normalized_data, scaler, data


class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder implemented with PyTorch.
    """

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Define encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Define decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # Pass through encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, dataloader, criterion, optimizer, epochs, device):
    """
    Trains the autoencoder on the given data.
    Args:
        model (torch.nn.Module): The autoencoder to train.
        dataloader (DataLoader): Data loader for training.
        criterion: Loss function (e.g., Mean Squared Error).
        optimizer: Optimizer for training (e.g., Adam).
        epochs (int): Number of epochs to train.
        device: Device ('cpu' or 'cuda') for computation.
    Returns:
        model: Trained autoencoder model.
        training_loss: List of training losses per epoch.
    """
    model.to(device)
    training_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_loss.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {training_loss[-1]:.4f}")
    return model, training_loss


def calculate_reconstruction_losses(model, data, device):
    """
    Calculates reconstruction loss for each sample in the dataset.
    Args:
        model (torch.nn.Module): Trained autoencoder.
        data (torch.Tensor): The input data to calculate losses for.
        device: Device ('cpu' or 'cuda') for computation.
    Returns:
        losses (numpy.ndarray): Array of reconstruction losses for each sample.
    """
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        reconstructed = model(data)
        losses = torch.mean((data - reconstructed) ** 2, dim=1).cpu().numpy()
    return losses


def visualize_reconstruction_loss(losses, threshold):
    """
    Visualizes reconstruction error distribution and the detection threshold.
    Args:
        losses (numpy.ndarray): Reconstruction losses of samples.
        threshold (float): Threshold for anomaly detection.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=50, alpha=0.7, label="Reconstruction Loss")
    plt.axvline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.3f})")
    plt.title("Reconstruction Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Settings
    file_path = "data/mock_data.csv"
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    threshold_multiplier = 3  # Multiplier for anomaly threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    data, scaler, original_data = preprocess_data(file_path)
    input_dim = data.shape[1]
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build and train autoencoder
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    autoencoder, training_loss = train_autoencoder(autoencoder, dataloader, criterion, optimizer, epochs, device)

    # Plot training loss
    plt.plot(training_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Calculate reconstruction losses
    full_data = torch.tensor(data, dtype=torch.float32)
    reconstruction_losses = calculate_reconstruction_losses(autoencoder, full_data, device)

    # Determine anomaly threshold
    threshold = np.mean(reconstruction_losses) + threshold_multiplier * np.std(reconstruction_losses)

    # Detect anomalies
    anomalies = np.where(reconstruction_losses > threshold)[0]
    print(f"Number of data points: {data.shape[0]}")
    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Indices of anomalies: {anomalies}")
    print(f"Threshold for anomaly detection: {threshold:.3f}")

    # Visualize reconstruction loss distribution
    visualize_reconstruction_loss(reconstruction_losses, threshold)

    # Save results to CSV
    output_df = original_data.copy()
    output_df["Reconstruction_Loss"] = reconstruction_losses
    output_df["Is_Anomaly"] = False
    output_df.loc[anomalies, "Is_Anomaly"] = True
    output_df.to_csv("data/anomaly_detection_autoencoder_pytorch.csv", index=False)
    print("\nResults saved to 'data/anomaly_detection_autoencoder_pytorch.csv'.")
