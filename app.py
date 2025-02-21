from dash import Dash, html, dcc

app = Dash(__name__)

app.layout = html.Div([
    html.H1('IoT Dashboard with Predictive Maintenance', style={'textAlign': 'center'}),
    dcc.Graph(
        id='temperature-chart',
        figure={
            "data": [
                {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "Device A"},
                {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "Device B"},
            ],
            "layout": {
                "title": "Example IoT Dashboard Graph"
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)