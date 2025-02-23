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
# Load Pre-Trained DBSCAN Model
dbscan_model = load_dbscan_model('data/models/dbscan_model.joblib')

# Load Clustered Data (if available)
try:
    clustered_data = pd.read_csv('data/detected_clusters.csv')
except FileNotFoundError:
    print("Error: 'data/detected_clusters.csv' not found.")
    clustered_data = pd.DataFrame(columns=['temperature', 'vibration', 'pressure', 'Cluster_Label', 'status'])

# Create Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "IoT Dashboard with Predictive Maintenance"

# Real-Time IoT Data (global variables for simulation)
iot_data = pd.DataFrame(columns=['timestamp', 'device_id', 'temperature', 'vibration', 'pressure'])

# ==========
# DASH LAYOUT
# ==========
app.layout = dbc.Container([
    # Header Row
    dbc.Row(
        dbc.Col(html.H1("IoT Dashboard with Predictive Maintenance", className="text-center text-primary mb-4"),
                width=12),
    ),

    # Main Layout Row: Sidebar + Main Content
    dbc.Row([
        # Sidebar (Left Column)
        dbc.Col([
            html.H4("Filters", className="text-secondary"),
            html.Hr(),
            html.Label("Select a Cluster to Visualize:", className="text-secondary"),
            dcc.Dropdown(
                id='cluster-selector',
                options=[{"label": f"Cluster {i}", "value": i} for i in clustered_data['Cluster_Label'].unique()],
                value=None,
                placeholder="Select a Cluster",
                className="mb-4"
            ),
            # Add more filtering options as needed
        ], width=3, style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "5px"}),

        # Main Content Area (Right Column)
        dbc.Col([
            # Real-Time Metrics Line Plot
            dbc.Card([
                dbc.CardHeader("Real-Time Metrics"),
                dbc.CardBody(dcc.Graph(id='live-metrics'))
            ], className="mb-4"),

            # 3D Cluster Visualization
            dbc.Card([
                dbc.CardHeader("3D Cluster Visualization"),
                dbc.CardBody(dcc.Graph(id='3d-cluster'))
            ], className="mb-4"),

            # Anomaly Count Bar Plot
            dbc.Card([
                dbc.CardHeader("Anomaly Count"),
                dbc.CardBody(dcc.Graph(id='anomaly-count'))
            ], className="mb-4"),
        ], width=9),
    ], className="mb-4"),
    dcc.Interval(
        id="update-interval",
        interval=1000,
        n_intervals=0
    )

], fluid=True)




# ==========
# CALLBACKS
# ==========
# Live Metrics Update
@app.callback(
    Output('live-metrics', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_live_metrics(n_intervals):
    global iot_data, clustered_data

    # Generate live IoT data
    new_data = generate_data(num_devices=5)  # Generate 5 new device readings
    iot_data = pd.concat([iot_data, new_data]).reset_index(drop=True)

    if dbscan_model:
        # Normalize the features for clustering
        features = preprocess_live_data(new_data)
        labels = predict_clusters_live(dbscan_model, features)

        # Add cluster predictions and status to the data
        new_data['Cluster_Label'] = labels
        new_data['status'] = new_data["Cluster_Label"].apply(lambda x: "Anomaly" if x == -1 else "Healthy")

        # Append to the clustered data and ensure consistency
        clustered_data = pd.concat([clustered_data, new_data]).reset_index(drop=True)

    # Create Real-Time Temperature Metrics Chart
    fig = px.line(
        iot_data[-50:],
        x='timestamp', y='temperature', color='device_id',
        title="Real-Time Temperature Metrics",
        labels={'timestamp': 'Time', 'temperature': 'Temperature (C)'},
        template="plotly_dark"
    )
    return fig


# 3D Clustering Visualization
@app.callback(
    Output('3d-cluster', 'figure'),
    [Input('cluster-selector', 'value')]
)
def update_cluster_viz(selected_cluster):
    global clustered_data

    # Filter data based on the selected cluster
    if selected_cluster is None:
        filtered_data = clustered_data
    else:
        filtered_data = clustered_data[clustered_data["Cluster_Label"] == selected_cluster]

    # Check if data is empty
    if filtered_data.empty:
        return px.scatter_3d(title="No Clusters to Display")

    # Ensure `status` column is available for color coding
    if 'status' not in filtered_data.columns:
        filtered_data['status'] = filtered_data["Cluster_Label"].apply(
            lambda x: "Anomaly" if x == -1 else "Healthy"
        )

    # Create 3D Cluster Visualization Plot
    fig = px.scatter_3d(
        filtered_data,
        x="temperature", y="vibration", z="pressure",
        color="status",  # Color by "Healthy" or "Anomaly"
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
    [Input('cluster-selector', 'value')]
)
def update_anomaly_count(selected_cluster):
    global clustered_data
    if clustered_data.empty:
        return px.bar(title="No Anomaly Data Available.")

    if selected_cluster is None:
        filtered_data = clustered_data
    else:
        filtered_data = clustered_data[clustered_data["Cluster_Label"] == selected_cluster]

    anomaly_count = filtered_data[filtered_data["status"] == "Anomaly"].groupby("Cluster_Label").size().reset_index(
        name="Count")
    if anomaly_count.empty:
        return px.bar(title="No Anomalies Found")

    fig = px.bar(
        anomaly_count,
        x="Cluster_Label", y="Count",
        title="Anomaly Count in Selected Clusters",
        labels={"Cluster_Label": "Cluster", "Count": "Number of Anomalies"},
    )
    return fig


# Dynamic Dropdown Update
@app.callback(
    Output('cluster-selector', 'options'),
    [Input('update-interval', 'n_intervals')]
)
def update_dropdown(n_intervals):
    global clustered_data
    unique_clusters = clustered_data['Cluster_Label'].unique()
    return [{"label": f"Cluster {i}", "value": i} for i in unique_clusters]


# Run the Dash App
if __name__ == '__main__':
    app.run_server(debug=True)
