import pandas as pd
import numpy as np


def generate_data(num_devices=3):
    """
    Generate real-time IoT data with anomalies (matching training data)
    """
    data = []
    for device_id in range(1, num_devices + 1):
        timestamp = pd.Timestamp.now()

        # Normal sensor readings (matching mock_data.csv)
        temperature = np.random.normal(loc=25.0, scale=2.0)
        vibration = np.random.normal(loc=3.0, scale=0.5)
        pressure = np.random.normal(loc=1.0, scale=0.1)

        is_anomaly = 0

        # Introduce anomalies with 20% probability (like training data)
        if np.random.rand() < 0.2:
            temperature += np.random.choice([-15, 15])  # Large temp spike/drop
            vibration += np.random.choice([-2, 2])  # Large vibration change
            pressure += np.random.choice([-0.5, 0.5])  # Pressure anomaly
            is_anomaly = 1
            print(f"Anomaly Created! Device {device_id} at {timestamp}")

        data.append({
            "timestamp": timestamp,
            "device_id": f"Device {device_id}",
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "is_anomaly": is_anomaly
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_data(num_devices=5)
    print(df)
