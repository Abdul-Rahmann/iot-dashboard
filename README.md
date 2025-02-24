# IoT Dashboard with Predictive Maintenance

## Project Overview
This project involves developing an **IoT Device Management Dashboard** using **Dash** to monitor device metrics in real-time, detect anomalies, and provide predictive maintenance capabilities. The dashboard incorporates machine learning models and IoT data visualization to ensure proactive monitoring and management of IoT devices.

---

## Features

### 1. **Real-Time IoT Device Metrics**
- Gather real-time data streams from IoT devices (e.g., temperature, CPU usage, vibration level, etc.).
- Visualize the incoming data using:
  - **Line Charts**: For sensor time-series data (e.g., temperature and vibration trends).
  - **Cluster Visualization (3D)**: Visualize device behavior patterns using DBSCAN clustering.
  - **Bar Charts**: Show anomaly counts and distributions across clusters.

---

### 2. **Device Status Monitoring**
- Display device statuses (e.g., **Healthy**, **Anomaly Detected**, or **Offline**) in real-time.
- Components:
  - **Detailed Device Status Table**:
    - Device IDs.
    - Current statuses (Healthy/Anomaly).
    - Key metrics (e.g., average temperature, last anomaly timestamp).
  - Dynamic notification system to alert users about anomalies.

---

### 3. **Predictive Maintenance**
- Use pre-trained ML models to predict device behavior and estimate failure risk.
- Features:
  - Train models using historical IoT data (e.g., device failure rates and sensor readings over time).
  - Display **Remaining Useful Life (RUL)** predictions:
    - Highlight devices with the lowest RUL values in red on the dashboard.

---

### 4. **Historical Data Visualization**
- Allow users to view long-term trends and aggregated metrics:
  - **Time-Series Views**:
    - Anomaly counts over time (e.g., per day or hour).
    - Sensor metric trends (temperature, pressure, etc.).
  - **Pie Charts**:
    - Distribution of anomalies by cluster or device.

---

### 5. **Device Interaction/Control**
- Add interactive controls to allow users to manage devices directly from the dashboard:
  - **Restart Devices**: Send commands to restart devices with anomaly states.
  - **Modify Parameters**: Enable users to adjust device settings (e.g., operating thresholds).

---

### 6. **Data Export and Reporting**
- Allow data to be exported in standard formats (e.g., CSV) for further analysis.
- Include options to download:
  - Real-time sensor metrics.
  - Anomaly detection results.
  - Cluster labels and associated details.

---

### 7. **IoT Simulator**
- Simulate an IoT data stream for dashboard testing and demo purposes.
- Features:
  - Generate realistic IoT metrics (temperature, pressure, vibration, etc.).
  - Introduce interdependencies (e.g., pressure affecting temperature).
  - Simulate device failures or anomalies.

---

## Architecture Overview

### 1. **Frontend (Dash)**
- Build a user-friendly dashboard interface with Dash.
- Use Plotly components for dynamic, interactive visualizations.
- Handle user interactions (e.g., device filtering, parameter control) with Dash callbacks.

### 2. **Backend (Flask or FastAPI)**
- Serve the real-time data stream to the dashboard.
- Handle API endpoints for device interactions (e.g., Restart Device commands).
- Store processed data in a database (optional).

### 3. **Predictive Maintenance ML Model**
- Utilize a **DBSCAN** (or similar) clustering model for real-time anomaly detection.
- Train additional ML models for:
  - Remaining Useful Life (RUL) predictions.
  - Device failure forecasting.
- Deploy these models using Flask/ONNX Runtime.

---

## Development Roadmap

### 1. **Phase 1: Basic Framework**
- Set up the Dash app and a simulated IoT data stream.
- Implement basic visualizations for real-time metrics.

### 2. **Phase 2: Anomaly Detection**
- Integrate pre-trained ML models (e.g., DBSCAN) for anomaly detection.
- Display anomaly alerts and notifications on the dashboard.

### 3. **Phase 3: Device Monitoring & Control**
- Add a detailed device status table with actionable controls (e.g., Restart/Modify Settings).

### 4. **Phase 4: Predictive Maintenance**
- Train and integrate an RUL prediction model.
- Visualize RUL estimates for devices.

### 5. **Phase 5: Data Export & Reporting**
- Allow users to save metrics, anomalies, and cluster information as CSV files.

### 6. **Phase 6: Historical Visualizations**
- Aggregate and display historical data and trends using long-term charts and summary visuals.

### 7. **Phase 7: IoT Simulator Enhancements**
- Make the simulator more realistic by adding:
  - Multi-sensor dependencies.
  - Device failure simulations.

---

## Setup Instructions

### Prerequisites
- Python 3.9 or newer.
- Required Python Packages:
  - `dash`, `dash-bootstrap-components`, `pandas`, `plotly`, `scikit-learn`, `numpy`.

### Steps
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Dash app with:
   ```bash
   python app.py
   ```
4. Open the dashboard in your browser at `http://127.0.0.1:8050`.

---

## Future Enhancements
- Add features for custom ML model training via the dashboard.
- Support additional IoT device types with more metrics (e.g., humidity, network traffic).
- Integrate WebSocket-based data streaming for higher responsiveness.
- Add user authentication for multi-user access control.

---
