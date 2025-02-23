import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import random
import pandas as pd

clustered_data = pd.read_csv('data/detected_clusters.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "IoT Dashboard with Predictive Maintenance"

app.layout = html.Div([
    html.H1("IoT Dashboard with Predictive Maintenance", style={"textAlign":"center"}),
    dcc.Graph(id='3d-cluster'),
    dcc.Graph(id='anomaly-count'),
    html.Div("Adjust the filters below", style={"marginTop":"20px"}),
    dcc.Dropdown(
        id='cluster-selector',
        options=[{"label":f"Cluster {i}","value":i} for i in clustered_data['Cluster_Label'].unique()],
        value=None,
        placeholder="Select a Cluster to Visualize",
    )
])

@app.callback(
    Output('3d-cluster', 'figure'),
    [Input('cluster-selector', 'value')]
)
def update_cluster_viz(selected_cluster):
    filtered_data = clustered_data if not selected_cluster else clustered_data[clustered_data['Cluster_Label'] == selected_cluster]
    fig = px.scatter_3d(
        filtered_data,
        x="temperature",
        y="vibration",
        z="pressure",
        color="Cluster_Label",
        title="3D Cluster Visualization",
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    return fig
@app.callback(
    Output('anomaly-count', 'figure'),
    [Input('cluster-selector', 'value')]
)
def update_anomaly_count(selected_cluster):
    filtered_data = clustered_data if not selected_cluster else clustered_data[
        clustered_data["Cluster_Label"] == selected_cluster]
    anomaly_count = filtered_data.groupby("Cluster_Label").size().reset_index(name="Count")

    fig = px.bar(
        anomaly_count,
        x="Cluster_Label",
        y="Count",
        title="Anomaly Count in Selected Cluster",
        labels={"Cluster_Label":"Cluster", "Count":"Anomaly Count"}
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)