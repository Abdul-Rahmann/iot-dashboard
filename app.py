import dash
import joblib
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from simulation.iot_simulator import generate_data
from ml.dbscan_anomaly_detection import preprocess_live_data, predict_clusters_live, load_dbscan_model

# ==============
# INITIAL SETUP
# ==============
# Load Pre-Trained DBSCAN Model & Scaler
dbscan_model = load_dbscan_model('data/models/dbscan_model.joblib')
scaler_path = "data/models/scaler.joblib"

# Create Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "IoT Dashboard with Predictive Maintenance"

# Real-Time IoT Data (global variable for simulation)
iot_data = pd.DataFrame(
    columns=['timestamp', 'device_id', 'temperature', 'vibration', 'pressure', 'Cluster_Label', 'status'])

# ==========
# DASH LAYOUT
# ==========
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("IoT Dashboard with Predictive Maintenance", className="text-center text-primary mb-4"),
                width=12),
    ),

    html.Div(id='anomaly-notification', className='mb-4'),

    dbc.Row([
        dbc.Col([
            html.H4("Filters", className="text-secondary"),
            html.Hr(),
            html.Label("Select a Cluster to Visualize:", className="text-secondary"),
            dcc.Dropdown(
                id='cluster-selector',
                options=[],  # Dynamically populated
                value=None,
                placeholder="Select a Cluster",
                className="mb-4"
            ),
        ], width=3, style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "5px"}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Real-Time Metrics"),
                dbc.CardBody(dcc.Graph(id='live-metrics'))
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("3D Cluster Visualization"),
                dbc.CardBody(dcc.Graph(id='3d-cluster'))
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Anomaly Count"),
                dbc.CardBody(dcc.Graph(id='anomaly-count'))
            ], className="mb-4"),
        ], width=9),
    ], className="mb-4"),

    dcc.Interval(
        id="update-interval",
        interval=2000,  # Increase interval to reduce processing load
        n_intervals=0
    )
], fluid=True)


# ==========
# CALLBACKS
# ==========

# Update Live Metrics
@app.callback(
    Output('live-metrics', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_live_metrics(n_intervals):
    global iot_data

    # Generate simulated IoT data
    new_data = generate_data(num_devices=3)

    # Ensure data is scaled using the same scaler as DBSCAN training
    features = preprocess_live_data(new_data, scaler_path="data/models/scaler.joblib")

    # Predict clusters using DBSCAN
    labels = predict_clusters_live(dbscan_model, features)
    new_data['Cluster_Label'] = labels
    new_data['status'] = new_data["Cluster_Label"].apply(lambda x: "Anomaly" if x == -1 else "Healthy")

    # Append new data to global dataframe
    iot_data = pd.concat([iot_data, new_data]).reset_index(drop=True)

    # Plot real-time temperature data
    fig = px.line(
        iot_data[-50:],  # Limit to last 50 records
        x='timestamp', y='temperature', color='device_id',
        title="Real-Time Temperature Metrics",
        labels={'timestamp': 'Time', 'temperature': 'Temperature (C)'},
        template="plotly_dark"
    )

    return fig


# Update Cluster Dropdown
@app.callback(
    Output('cluster-selector', 'options'),
    [Input('update-interval', 'n_intervals')]
)
def update_dropdown(n_intervals):
    if iot_data.empty:
        return []

    unique_clusters = iot_data['Cluster_Label'].dropna().unique()

    return [{"label": f"Cluster {i}", "value": i} for i in unique_clusters]


# 3D Cluster Visualization
@app.callback(
    Output('3d-cluster', 'figure'),
    [Input('cluster-selector', 'value')]
)
def update_cluster_viz(selected_cluster):
    if iot_data.empty:
        return px.scatter_3d(title="No Data Available")

    # Filter by selected cluster
    filtered_data = iot_data if selected_cluster is None else iot_data[iot_data["Cluster_Label"] == selected_cluster]

    if filtered_data.empty:
        return px.scatter_3d(title="No Clusters to Display")

    # Ensure 'status' column exists
    if 'status' not in filtered_data.columns:
        filtered_data['status'] = filtered_data["Cluster_Label"].apply(lambda x: "Anomaly" if x == -1 else "Healthy")

    fig = px.scatter_3d(
        filtered_data,
        x="temperature", y="vibration", z="pressure",
        color="status",
        symbol="device_id",
        title="3D Cluster Visualization",
        labels={"status": "Status"},
        color_discrete_map={"Healthy": "green", "Anomaly": "red"},
        template="plotly_dark"
    )
    return fig


# Anomaly Count Visualization
@app.callback(
    Output('anomaly-count', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_anomaly_count(n_intervals):
    if iot_data.empty:
        return px.bar(title="No Anomaly Data Available.")

    # Count anomalies per cluster
    anomaly_count = iot_data[iot_data["status"] == "Anomaly"].groupby("Cluster_Label").size().reset_index(name="Count")

    if anomaly_count.empty:
        return px.bar(title="No Anomalies Found")

    fig = px.bar(
        anomaly_count,
        x="Cluster_Label", y="Count",
        title="Anomaly Count in Selected Clusters",
        labels={"Cluster_Label": "Cluster", "Count": "Number of Anomalies"},
    )
    return fig


# Run Dash App
if __name__ == '__main__':
    app.run_server(debug=True)