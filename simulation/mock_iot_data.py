import pandas as pd
import numpy as np
import time

def generate_iot_data(num_devices=5, num_samples=100, interval=1.0):
    """
    Simulates IoT sensor data with time.
    Args:
        num_devices: Number of devices generating data.
        num_samples: Number of samples per device.
        interval: Seconds between each sample generation (for real-time simulation).

    Returns:
        DataFrame with simulated time-series sensor data.
    """
    data = []
    for i in range(num_devices):
        timestamp = pd.Timestamp.now()
        for device_id in range(1, num_samples + 1):
            temperature = np.random.normal(loc=25.0, scale=2.0)
            vibration = np.random.normal(loc=3.0, scale=0.5)
            pressure = np.random.normal(loc=1, scale=0.1)

            if i > num_samples // 2 and device_id == 3:  # Simulate anomalies for device 3
                temperature += np.random.choice([-15, 15])  # Add a large spike/drop
                vibration += np.random.choice([-2, 2])  # Spike or drop in vibration
                pressure += np.random.choice([-0.5, 0.5])  # Simulate pressure anomaly

            data.append([timestamp, device_id, temperature, vibration, pressure])
        time.sleep(interval)

    columns = ['timestamp', 'device_id', 'temperature', 'vibration', 'pressure']

    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    mock_data = generate_iot_data(num_devices=3, num_samples=50, interval=0.1)
    print(mock_data.head())
    mock_data.to_csv('data/mock_data.csv', index=False)
    print("Sample data saved to data/iot_data.csv")