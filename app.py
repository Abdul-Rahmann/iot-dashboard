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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "IoT Dashboard with Predictive Maintenance"

# Real-Time IoT Data (global variable for simulation)
iot_data = pd.DataFrame(
    columns=['timestamp', 'device_id', 'temperature', 'vibration', 'pressure', 'Cluster_Label', 'status'])

# ==========
# DASH LAYOUT
# ==========

# Enhanced Layout
app.layout = dbc.Container([
    # Title Row
    dbc.Row(
        dbc.Col(html.H1("IoT Dashboard with Predictive Maintenance",
                        className="text-center text-light mb-4"), width=12)
    ),

    # KPIs Section
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Devices Online", className="text-success"),
                html.H2(id="kpi-devices-online", children="0", className="text-success")
            ])
        ], className="card border-success mb-4"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Anomalies", className="text-danger"),
                html.H2(id="kpi-total-anomalies", children="0", className="text-danger")
            ])
        ], className="card border-danger mb-4"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Avg. Temperature (°C)", className="text-info"),
                html.H2(id="kpi-average-temp", children="0", className="text-info")
            ])
        ], className="card border-info mb-4"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Critical Devices", className="text-warning"),
                html.H2(id="kpi-critical-devices", children="0", className="text-warning")
            ])
        ], className="card border-warning mb-4"), width=3),
    ]),

    # Sidebar and Main Content Row
    dbc.Row([
        # Sidebar (Filters Section)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Filters", className="text-primary")),
                dbc.CardBody([
                    html.Label("Select a Cluster:", className="text-primary"),
                    dcc.Dropdown(
                        id='cluster-selector',
                        options=[],  # Dynamically populated
                        value=None,
                        placeholder="Select a Cluster",
                        className="mb-3"
                    ),
                    html.Label("Critical RUL Threshold:", className="text-primary"),
                    dcc.Slider(
                        id="rul-threshold-slider",
                        min=1, max=50, step=1, value=10,
                        marks={i: str(i) for i in range(1, 51, 5)},
                        className="mb-4"
                    ),
                    html.Label("Update Interval (Seconds):", className="text-primary"),
                    dcc.Input(
                        id="update-interval-input",
                        type="number", min=1, step=1, value=10,
                        className="mb-3"
                    ),
                    dbc.Button("Update Filters", id="filter-button", color="light", class_name="w-100")
                ])
            ], className="mb-4 text-light"),
        ], width=3, style={"backgroundColor": "#343a40", "padding": "20px", "borderRadius": "8px"}),

        # Main Content
        dbc.Col([
            # Real-Time Metrics and Charts Row
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H5("Live Metrics", className="text-center text-info")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-live-metrics",
                            type="default",
                            children=dcc.Graph(id='live-metrics')
                        )
                    ])
                ], className="mb-4"), width=6),

                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H5("3D Cluster Visualization", className="text-center text-info")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-cluster-visualization",
                            type="circle",
                            children=dcc.Graph(id='3d-cluster')
                        )
                    ])
                ], className="mb-4"), width=6)
            ]),

            # Anomaly and RUL Charts Row
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H5("Anomaly Count", className="text-center text-warning")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-anomaly-chart",
                            type="circle",
                            children=dcc.Graph(id='anomaly-count')
                        )
                    ])
                ], className="mb-4"), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H5("Remaining Useful Life (RUL)", className="text-center text-danger")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-rul-chart",
                            type="circle",
                            children=dcc.Graph(id='rul-bar-chart')
                        )
                    ])
                ], className="mb-4"), width=6)
            ]),

            # Alerts Section
            dbc.Row([
                dbc.Col(
                    html.Div(id="critical-devices-alert", className="alert alert-danger text-center"),
                    width=12
                )
            ]),
        ], width=9)
    ]),

    # Device Status Table Section
    dbc.Row(
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Device Status Table", className="text-success")),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-table",
                    type="circle",
                    children=dash.dash_table.DataTable(
                        id='device-status-table',
                        columns=[
                            {'name': 'Device ID', 'id': 'device_id'},
                            {'name': 'Average Temperature', 'id': 'avg_temperature'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Last Anomaly Timestamp', 'id': 'last_anomaly'},
                            {'name': 'Predicted RUL', 'id': 'predicted_rul'},
                            {'name': 'RUL Status', 'id': 'rul_status'}
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'backgroundColor': '#393939',
                            'color': 'white'
                        },
                        style_header={'backgroundColor': '#6c757d', 'fontWeight': 'bold', 'color': 'white'},
                    )
                )
            ]),
        ], className="mb-4"), width=12)
    ),

    # CSV Download Section
    dbc.Row(
        dbc.Col(dbc.Card([
            dbc.CardBody([
                dbc.Button("Download CSV", id="download-button", color="primary", className="me-2"),
                dcc.Download(id="download-dataframe-csv")
            ])
        ], className="mb-4"), width=12)
    ),

    # Interval Component
    dcc.Interval(
        id="update-interval",
        interval=2000,
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
# Callback for Updating KPIs
@app.callback(
    [
        Output("kpi-devices-online", "children"),
        Output("kpi-total-anomalies", "children"),
        Output("kpi-average-temp", "children"),
        Output("kpi-critical-devices", "children")
    ],
    [Input("update-interval", "n_intervals"),
     Input("rul-threshold-slider", "value")
     ]
)
def update_kpis(n_intervals, rul_threshold):
    if iot_data.empty:
        return 0, 0, 0, 0  # Default values when no data

    # KPI 1: Total Devices Online
    total_devices = len(iot_data['device_id'].unique())

    # KPI 2: Total Anomalies
    total_anomalies = iot_data[iot_data["status"] == "Anomaly"].shape[0]

    # KPI 3: Average Temperature
    avg_temp = round(iot_data["temperature"].mean(), 2) if not iot_data.empty else 0

    # KPI 4: Critical Devices (RUL <= threshold)
    # rul_threshold = 10  # Can be taken from user input
    device_ids = iot_data['device_id'].unique()
    rul_predictions = simulate_rul_predictions(device_ids)
    critical_devices = sum([1 for pred in rul_predictions if pred["predicted_rul"] <= rul_threshold])

    return total_devices, total_anomalies, avg_temp, critical_devices


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
    [Input("update-interval", "n_intervals"),
     Input("rul-threshold-slider", "value")
     ]
)
def update_rul_predictions(n_intervals, rul_threshold):
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
    fig.add_hline(y=rul_threshold, line_dash="dot", line_color="red", annotation_text="Maintenance Needed")

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
