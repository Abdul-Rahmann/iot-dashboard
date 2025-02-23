import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import random

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "IoT Dashboard with Predictive Maintenance"

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("IoT Dashboard with Predictive Maintenance", className="text-center"), width=12)),
    dbc.Row(dbc.Col(dcc.Graph(id='live-graph', style={'height': '60vh'}), width=12)),

    dcc.Interval(
        id='update-interval',
        interval=1000,
        n_intervals=0
    ),
],fluid=True)

@app.callback(
    Output( 'live-graph', 'figure' ),
    Input( 'update-interval', 'n_intervals')
)
def update_graph(n_intervals):
    device_ids = ["Device 1", "Device 2", "Device 3"]
    metrics = [random.randint(20, 100) for _ in device_ids]

    figure = go.Figure(
        data=[go.Bar(x=device_ids, y=metrics)]
    )

    figure.update_layout(
        title='Real-Time Device Metrics',
        xaxis={"title": "Devices"},
        yaxis={"title": "Metric Readings"}
    )

    return figure

if __name__ == '__main__':
    app.run_server(debug=True)