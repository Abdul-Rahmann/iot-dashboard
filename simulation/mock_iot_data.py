import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt


def generate_iot_data(num_devices=5, num_samples=100, interval=0.1):
    """
    Simulates IoT sensor data with time.
    """
    data = []

    for i in range(num_devices):
        for device_id in range(1, num_samples + 1):
            timestamp = pd.Timestamp.now() + pd.Timedelta(seconds=device_id * 0.01)  # Unique timestamps

            # Normal sensor readings
            temperature = np.random.normal(loc=25.0, scale=2.0)
            vibration = np.random.normal(loc=3.0, scale=0.5)
            pressure = np.random.normal(loc=1, scale=0.1)

            is_anomaly = 0

            # Introduce anomalies randomly across all devices
            if np.random.rand() < 0.2:  # 20% probability for any device
                temperature += np.random.choice([-15, 15])  # Large temperature spike/drop
                vibration += np.random.choice([-2, 2])  # Large vibration change
                pressure += np.random.choice([-0.5, 0.5])  # Pressure anomaly
                is_anomaly = 1
                print(f"Anomaly Created! Device {device_id} at {timestamp}")  # Debugging

            data.append([timestamp, device_id, temperature, vibration, pressure, is_anomaly])

        time.sleep(interval)

    columns = ['timestamp', 'device_id', 'temperature', 'vibration', 'pressure', 'is_anomaly']
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    mock_data = generate_iot_data(num_devices=3, num_samples=50, interval=0.1)
    mock_data.to_csv('data/mock_data.csv', index=False)

    print("\nAnomaly Counts:\n", mock_data["is_anomaly"].value_counts())
    print("\nAnomalies in dataset:\n", mock_data[mock_data["is_anomaly"] == 1])  # Print anomalies

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    features = ["temperature", "vibration", "pressure"]

    for i, feature in enumerate(features):
        sns.histplot(mock_data[feature], kde=True, ax=axes[i], label="All Data", alpha=0.6)

        if "is_anomaly" in mock_data.columns and mock_data["is_anomaly"].sum() > 0:
            sns.histplot(mock_data[mock_data["is_anomaly"] == 1][feature], kde=True, ax=axes[i],
                         color="red", label="Anomalies", alpha=0.8)

        axes[i].set_title(f"Distribution of {feature}")
        axes[i].legend()

    plt.tight_layout()
    plt.show()