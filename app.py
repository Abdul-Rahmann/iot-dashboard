import dash
import joblib
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import io
import random
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
    html.Div(id="critical-devices-alert", className="mb-4"),

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

        dbc.Card([
            dbc.CardBody([
                html.Label("Critical RUL Threshold:"),
                dcc.Slider(
                    id="rul-threshold-slider",
                    min=1,
                    max=50,
                    step=1,
                    value=10,  # Default threshold
                    marks={i: str(i) for i in range(1, 51, 5)}
                ),
                html.Br(),
                html.Label("Update Interval (Seconds):"),
                dcc.Input(
                    id="update-interval-input",
                    type="number",
                    min=1,
                    value=10,  # Default interval
                    step=1
                )
            ])
        ], className="mb-4"),

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

    dbc.Card([
        dbc.CardHeader("Remaining Useful Life (RUL) Prediction"),
        dbc.CardBody([
            dcc.Loading(
                id="loading-rul-chart",
                type="circle",
                children=[
                    dcc.Graph(id="rul-bar-chart")
                ]
            )
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Device Status Table"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-table",
                type="circle",
                children=[
                    dash.dash_table.DataTable(
                        id='device-status-table',
                        columns=[
                            {'name': 'Device ID', 'id': 'device_id'},
                            {'name': 'Average Temperature', 'id': 'avg_temperature'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Last Anomaly Timestamp', 'id': 'last_anomaly'},
                            {'name': 'Predicted RUL', 'id': 'predicted_rul'},  # Add RUL column
                            {'name': 'RUL Status', 'id': 'rul_status'},  # Add Health Status column
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    )
                ]
            )
        )
    ], className="mb-4"),

    # Add to the layout
    dbc.Card([
        dbc.CardBody([
            dbc.Button("Download CSV", id="download-button", color="primary", className="me-2"),
            dcc.Download(id="download-dataframe-csv")
        ])
    ], className="mb-4"),

    dcc.Interval(
        id="update-interval",
        interval=2000,  # Increase interval to reduce processing load
        n_intervals=0
    )
], fluid=True)


# =================
# HELPER FUNCTIONS
# =================
def simulate_rul_predictions(devices):
    """
    Simulate Remaining Useful Life (RUL) predictions for the given devices.
    Devices with a status of 'Anomaly' will have lower RUL.
    """
    rul_predictions = []
    for device_id in devices:
        # Generate RUL between 5-50 days for "Healthy" and 1-10 days for "Anomaly"
        if iot_data[iot_data['device_id'] == device_id].iloc[-1]["status"] == "Anomaly":
            rul = random.randint(1, 10)  # Low RUL due to anomaly
        else:
            rul = random.randint(10, 50)  # Higher RUL if Healthy

        rul_predictions.append({'device_id': device_id, 'predicted_rul': rul})

    return rul_predictions


# ==========
# CALLBACKS
# ==========
@app.callback(
    Output("critical-devices-alert", "children"),
    [Input("update-interval", "n_intervals")]
)
def update_critical_devices_alert(n_intervals):
    if iot_data.empty:
        return None

    # Simulate predictions
    device_ids = iot_data['device_id'].unique()
    rul_predictions = simulate_rul_predictions(device_ids)

    # Identify critical devices
    critical_devices = [pred['device_id'] for pred in rul_predictions if pred['predicted_rul'] <= 10]

    if not critical_devices:
        return html.Div("All devices are in good condition.", className="alert alert-success")

    # Create an alert for critical devices
    return html.Div(
        f"Warning: The following devices have critically low RUL: {', '.join(critical_devices)}.",
        className="alert alert-danger"
    )


@app.callback(
    Output("rul-bar-chart", "figure"),
    [Input("update-interval", "n_intervals")]
)
def update_rul_predictions(n_intervals):
    if iot_data.empty:
        # Return an empty figure if no data is available
        return px.bar(title="No RUL Data Available")

    # Get list of device IDs
    device_ids = iot_data['device_id'].unique()

    # Simulate RUL predictions for devices
    rul_predictions = simulate_rul_predictions(device_ids)

    # Convert to a DataFrame for visualization
    rul_df = pd.DataFrame(rul_predictions)

    # Create a bar chart (device_id vs predicted_rul)
    fig = px.bar(
        rul_df, x="device_id", y="predicted_rul",
        labels={"predicted_rul": "RUL (Days)", "device_id": "Device ID"},
        title="Remaining Useful Life (RUL) by Device",
        template="plotly_white",
        color="predicted_rul",  # Optional color coding by RUL
        color_continuous_scale="Viridis"
    )

    # Add threshold line (RUL < 10 days is an alert)
    fig.add_hline(y=10, line_dash="dot", line_color="red", annotation_text="Maintenance Needed")

    return fig


@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-button", "n_clicks")],
    prevent_initial_call=True  # Ensures callback only runs after the button is clicked
)
def export_csv(n_clicks):
    # Ensure there's data to export
    if iot_data.empty:
        return dcc.send_file(io.StringIO("No data available"), filename="iot_data.csv")

    # Convert the iot_data DataFrame into a CSV buffer
    buffer = io.StringIO()
    export_data = iot_data.copy()
    export_data.to_csv(buffer, index=False)

    # Reset the buffer for downloading
    buffer.seek(0)

    # Provide the data for download
    return dcc.send_data_frame(export_data.to_csv, filename="iot_data.csv")


@app.callback(
    Output('device-status-table', 'data'),
    [Input('update-interval', 'n_intervals'),
     Input("rul-threshold-slider", "value")
     ]
)
def update_device_status_table(n_intervals, rul_threshold):
    if iot_data.empty:
        return []

    # Aggregate data: Average metrics and status per device
    device_data = iot_data.groupby('device_id').agg({
        'temperature': 'mean',  # Average temperature
        'timestamp': 'max',  # Most recent timestamp
        'status': lambda x: x.iloc[-1]  # Latest status
    }).reset_index()

    # Simulate RUL predictions for devices
    rul_predictions = simulate_rul_predictions(device_data['device_id'].values)
    rul_df = pd.DataFrame(rul_predictions)

    # Merge RUL predictions with device metrics
    device_data = device_data.merge(rul_df, on='device_id', how='left')

    # Add a critical alert column
    device_data['rul_status'] = device_data['predicted_rul'].apply(
        lambda rul: "Critical" if rul <= rul_threshold else "Healthy"
    )

    # Rename columns for display
    device_data.rename(columns={
        'temperature': 'avg_temperature',
        'timestamp': 'last_anomaly',
    }, inplace=True)

    # Convert to list of dictionaries for DataTable
    return device_data.to_dict('records')


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

    iot_data = iot_data.tail(500)

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
