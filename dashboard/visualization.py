import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import sqlite3

DB_PATH = r"C:/Hackathon Projects/SIH Project/bus_data.db"

def fetch_from_db(query, params=None):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn, params=params)

def get_routes():
    # Only offer routes that are seen in both vehicle_positions and routes table
    route_set = set(fetch_from_db("SELECT DISTINCT route_id FROM vehicle_positions")['route_id'].astype(str))
    df = fetch_from_db("SELECT route_id, route_short_name, route_long_name FROM routes")
    opts = []
    for _, r in df.iterrows():
        rid = str(r.route_id)
        if rid in route_set:
            label = f"{rid}"
            if isinstance(r.route_short_name, str) and r.route_short_name.strip():
                label += f" ({r.route_short_name.strip()})"
            opts.append({'label': label, 'value': rid})
    return opts

def get_vehicle_positions(route_id=None, time_window_hours=4):
    now = int(datetime.utcnow().timestamp())
    since = now - time_window_hours * 3600
    if route_id and route_id != 'all':
        query = """
        SELECT * FROM vehicle_positions
        WHERE route_id = ? AND timestamp >= ?
        ORDER BY timestamp
        """
        return fetch_from_db(query, (route_id, since))
    else:
        query = """
        SELECT * FROM vehicle_positions
        WHERE timestamp >= ?
        ORDER BY timestamp
        """
        return fetch_from_db(query, (since,))

def get_stops(route_id):
    # Note: trips.route_id is INTEGER, but vehicle_positions.route_id and our dropdown returns as STRING.
    try:
        rid_int = int(route_id)
    except Exception:
        return pd.DataFrame()
    stops_query = """
        SELECT DISTINCT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
        FROM stops s
        JOIN stop_times st ON s.stop_id = st.stop_id
        JOIN trips t ON t.trip_id = st.trip_id
        WHERE t.route_id = ?
        ORDER BY s.stop_id
    """
    return fetch_from_db(stops_query, (rid_int,))

def calculate_headways(timestamps):
    return [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if t2 > t1]

class BusSystemDashboard:
    def __init__(self):
        external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"]
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.title = "Bus Service Rationalization Dashboard"
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        # Use first available route (if available), else 'all'
        route_opts = get_routes()
        default_route = route_opts[0]['value'] if route_opts else 'all'
        self.app.layout = html.Div([
            html.Div([
                html.Div([
                    html.H1("Bus Service Rationalization Dashboard", className="fw-bold display-5 mb-3 text-center"),
                    html.P("Real-time monitoring of bus system performance and control strategies",
                           className="fs-5 text-center mb-5 text-muted"),
                ], className="col-12"),
            ], className="row bg-white py-5 mb-5 shadow-sm rounded-4 border-bottom"),
            
            html.Div([
                html.Div([
                    html.Label("Route Selection:", className="form-label fw-semibold text-muted mb-3"),
                    dcc.Dropdown(
                        id='route-selector',
                        options=route_opts + [{'label': 'All Routes', 'value': 'all'}],
                        value=default_route,
                        className="mb-4"
                    )
                ], className="col-md-4 mb-4"),
                html.Div([
                    html.Label("Time Window:", className="form-label fw-semibold text-muted mb-3"),
                    dcc.Dropdown(
                        id='time-window',
                        options=[
                            {'label': 'Last Hour', 'value': 1},
                            {'label': 'Last 4 Hours', 'value': 4},
                            {'label': 'Last 24 Hours', 'value': 24}
                        ],
                        value=4,
                        className="mb-4"
                    )
                ], className="col-md-4 mb-4"),
                html.Div([
                    html.Label("Refresh Data", className="form-label fw-semibold text-muted mb-3"),
                    html.Button("Refresh Data", id="refresh-btn", className="btn btn-dark w-100 rounded-pill py-3"),
                ], className="col-md-4 mb-4"),
            ], className="row mb-5 g-4"),

            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Service Regularity", className="card-title fw-semibold text-dark mb-2"),
                        html.H2(id="headway-cv", className="card-text fw-bold mb-2"),
                        html.P("Coefficient of Variation", className="text-muted mb-0 fs-6")
                    ], className="card-body p-5")
                ], className="col-md-3 mb-4 card shadow-sm border-0 rounded-4"),
                html.Div([
                    html.Div([
                        html.H5("Bunching Rate", className="card-title fw-semibold text-dark mb-2"),
                        html.H2(id="bunching-rate", className="card-text fw-bold mb-2"),
                        html.P("% of irregular headways", className="text-muted mb-0 fs-6")
                    ], className="card-body p-5")
                ], className="col-md-3 mb-4 card shadow-sm border-0 rounded-4"),
                html.Div([
                    html.Div([
                        html.H5("Avg Wait Time", className="card-title fw-semibold text-dark mb-2"),
                        html.H2(id="wait-time", className="card-text fw-bold mb-2"),
                        html.P("Minutes", className="text-muted mb-0 fs-6")
                    ], className="card-body p-5")
                ], className="col-md-3 mb-4 card shadow-sm border-0 rounded-4"),
                html.Div([
                    html.Div([
                        html.H5("Control Actions", className="card-title fw-semibold text-dark mb-2"),
                        html.H2(id="control-actions", className="card-text fw-bold mb-2"),
                        html.P("Holds applied", className="text-muted mb-0 fs-6"),
                    ], className="card-body p-5"),
                ], className="col-md-3 mb-4 card shadow-sm border-0 rounded-4"),
            ], className="row mb-5 g-4"),

            html.Div([
                html.Div([dcc.Graph(id="headway-distribution")], className="col-md-6 mb-4 card shadow-sm border-0 rounded-4 p-5"),
                html.Div([dcc.Graph(id="real-time-headways")], className="col-md-6 mb-4 card shadow-sm border-0 rounded-4 p-5"),
            ], className="row mb-5 g-4"),

            html.Div([
                html.Div([dcc.Graph(id="route-map")], className="col-md-6 mb-4 card shadow-sm border-0 rounded-4 p-5"),
            ], className="row mb-5 g-4"),

            dcc.Interval(id='interval-component', interval=30 * 1000, n_intervals=0),
            dcc.Store(id='dashboard-data')
        ], className="container-xl py-5 bg-white")

    def _setup_callbacks(self):
        @self.app.callback(
            Output('dashboard-data', 'data'),
            [Input('interval-component', 'n_intervals'), Input('refresh-btn', 'n_clicks')],
            [State('route-selector', 'value'), State('time-window', 'value')]
        )
        def update_dashboard_data(n_intervals, n_clicks, route_id, time_window_hours):
            try:
                df = get_vehicle_positions(route_id, time_window_hours)
                if not df.empty and 'timestamp' in df:
                    df = df.sort_values('timestamp')
                    bus_positions = [
                        {'id': row['vehicle_id'], 'lat': row['latitude'], 'lon': row['longitude']}
                        for _, row in df.iterrows() if pd.notnull(row['latitude']) and pd.notnull(row['longitude'])
                    ]
                    timestamps = df['timestamp'].tolist()
                    headways = calculate_headways(timestamps)
                    stops = get_stops(route_id) if route_id != 'all' else pd.DataFrame()
                else:
                    bus_positions, headways, stops = [], [], pd.DataFrame()
                time_series = {
                    'timestamps': [datetime.utcfromtimestamp(ts).strftime('%H:%M:%S') for ts in df['timestamp'].tolist()] if not df.empty else [],
                    'headways': headways
                }
                # Metrics calculation
                cv = (pd.Series(headways).std() / pd.Series(headways).mean()) if headways and pd.Series(headways).mean() > 0 else 0
                bunching = sum(1 for h in headways if h < 300) / len(headways) if headways else 0  # 300 = 50% of target headway
                avg_wait_time = (sum([h**2 for h in headways]) / (2 * sum(headways))) if headways and sum(headways) > 0 else 0
                # No control actions from DB, set as 0
                controls = 0

                return {
                    'bus_positions': bus_positions,
                    'headways': headways,
                    'headway_cv': cv,
                    'bunching_rate': bunching,
                    'avg_wait_time': avg_wait_time,
                    'control_actions': controls,
                    'time_series': time_series,
                    'stops': stops.to_dict('records') if not stops.empty else []
                }
            except Exception as e:
                print(f"Error updating data: {e}")
                return {}

        @self.app.callback(
            [Output('headway-cv', 'children'),
             Output('bunching-rate', 'children'),
             Output('wait-time', 'children'),
             Output('control-actions', 'children')],
            [Input('dashboard-data', 'data')]
        )
        def update_kpis(data):
            if not data:
                return "N/A", "N/A", "N/A", "N/A"
            cv = data.get('headway_cv', 0)
            bunching = data.get('bunching_rate', 0) * 100
            wait_time = data.get('avg_wait_time', 0) / 60
            controls = data.get('control_actions', 0)
            return f"{cv:.3f}", f"{bunching:.1f}%", f"{wait_time:.1f}", f"{controls}"

        @self.app.callback(
            Output('headway-distribution', 'figure'),
            [Input('dashboard-data', 'data')]
        )
        def update_headway_distribution(data):
            if not data or 'headways' not in data or not data['headways']:
                return go.Figure()
            headways = data['headways']
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=headways,
                nbinsx=20,
                name="Headway Distribution",
                opacity=0.7
            ))
            target_headway = 600
            fig.add_vline(x=target_headway, line_dash="dash", annotation_text="Target", line_color="red")
            fig.update_layout(
                title="Headway Distribution",
                xaxis_title="Headway (seconds)",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            return fig

        @self.app.callback(
            Output('real-time-headways', 'figure'),
            [Input('dashboard-data', 'data')]
        )
        def update_realtime_headways(data):
            if not data or 'time_series' not in data or not data['time_series']['headways']:
                return go.Figure()
            ts_data = data['time_series']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_data['timestamps'],
                y=ts_data['headways'],
                mode='lines+markers',
                name='Actual Headways',
                line=dict(color='blue')
            ))
            fig.add_hline(y=600, line_dash="dash", annotation_text="Target", line_color="red")
            fig.update_layout(
                title="Real-time Headway Monitoring",
                xaxis_title="Time",
                yaxis_title="Headway (seconds)",
                template="plotly_white"
            )
            return fig

        @self.app.callback(
            Output('route-map', 'figure'),
            [Input('dashboard-data', 'data')]
        )
        def update_route_map(data):
            if not data or 'bus_positions' not in data or not data['bus_positions']:
                return go.Figure()
            bus_positions = data['bus_positions']
            fig = go.Figure()
            fig.add_trace(go.Scattermapbox(
                lat=[pos['lat'] for pos in bus_positions],
                lon=[pos['lon'] for pos in bus_positions],
                mode='markers',
                marker=dict(size=12, color='blue'),
                text=[f"Bus {pos['id']}" for pos in bus_positions],
                name='Buses'
            ))
            if 'stops' in data and data['stops']:
                stops = data['stops']
                fig.add_trace(go.Scattermapbox(
                    lat=[stop['stop_lat'] for stop in stops],
                    lon=[stop['stop_lon'] for stop in stops],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='circle'),
                    text=[stop['stop_name'] for stop in stops],
                    name='Stops'
                ))
            fig.update_layout(
                title="Real-time Bus Positions",
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=28.6139, lon=77.2090),
                    zoom=12
                ),
                height=400
            )
            return fig

    def run(self, host='127.0.0.1', port=8050, debug=True):
        print(f"Starting Bus System Dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    dashboard = BusSystemDashboard()
    print("Starting Bus System Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser")
    dashboard.run()
