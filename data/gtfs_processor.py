import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sqlite3
import requests
import zipfile
import io
import os
from datetime import datetime, timedelta
import logging

class GTFSProcessor:
    """
    Processes static GTFS data from Delhi OTD portal and creates
    the digital network representation for the bus system.
    """

    def __init__(self, database_url: str = "sqlite:///bus_data.db"):
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)

        # Core GTFS tables
        self.stops = None
        self.routes = None 
        self.trips = None
        self.stop_times = None
        self.shapes = None

        # Track data source
        self.data_source = None  # "mock" or "realtime"

    def download_gtfs_data(self, gtfs_url: str) -> bool:
        """Download GTFS data from Delhi OTD portal"""
        try:
            response = requests.get(gtfs_url)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extractall("gtfs_data/")

            self.logger.info("GTFS data downloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download GTFS data: {e}")
            return False

    def load_gtfs_files(self, gtfs_dir: str = "gtfs_data/") -> bool:
        """Load GTFS CSV files into pandas DataFrames"""
        try:
            # Required GTFS files
            required_files = {
                'stops': 'stops.txt',
                'routes': 'routes.txt', 
                'trips': 'trips.txt',
                'stop_times': 'stop_times.txt'
            }

            # Optional files
            optional_files = {
                'shapes': 'shapes.txt',
                'calendar': 'calendar.txt',
                'calendar_dates': 'calendar_dates.txt'
            }

            # Load required files
            all_found = True
            for attr, filename in required_files.items():
                filepath = os.path.join(gtfs_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, attr, pd.read_csv(filepath))
                    self.logger.info(f"Loaded {filename}")
                else:
                    self.logger.error(f"Required file {filename} not found")
                    all_found = False

            # Load optional files
            for attr, filename in optional_files.items():
                filepath = os.path.join(gtfs_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, attr, pd.read_csv(filepath))
                    self.logger.info(f"Loaded {filename}")

            if all_found:
                self.data_source = "realtime"
                self.logger.info("GTFSProcessor: Proceeded with real GTFS data.")
            else:
                self.data_source = "mock"
                self.logger.warning("GTFSProcessor: Proceeded with mock data due to missing required files.")

            return all_found

        except Exception as e:
            self.logger.error(f"Failed to load GTFS files: {e}")
            self.data_source = "mock"
            return False

    def create_route_graph(self, route_id: str) -> Dict:
        """Create a graph representation of a specific route"""
        try:
            # Get trips for the route
            route_trips = self.trips[self.trips['route_id'] == route_id]

            if route_trips.empty:
                self.logger.warning(f"No trips found for route {route_id}")
                return {}

            # Get stop sequence for the route (using first trip)
            sample_trip = route_trips.iloc[0]['trip_id']
            trip_stops = self.stop_times[
                self.stop_times['trip_id'] == sample_trip
            ].sort_values('stop_sequence')

            # Create ordered list of stops
            stop_sequence = trip_stops['stop_id'].tolist()

            # Get stop details
            route_stops = self.stops[self.stops['stop_id'].isin(stop_sequence)]

            # Create route graph
            route_graph = {
                'route_id': route_id,
                'stop_sequence': stop_sequence,
                'stops': route_stops.to_dict('records'),
                'segments': []
            }

            # Create segments between consecutive stops
            for i in range(len(stop_sequence) - 1):
                current_stop = stop_sequence[i]
                next_stop = stop_sequence[i + 1]

                # Get stop coordinates
                current_coords = route_stops[route_stops['stop_id'] == current_stop]
                next_coords = route_stops[route_stops['stop_id'] == next_stop]

                if not current_coords.empty and not next_coords.empty:
                    segment = {
                        'from_stop': current_stop,
                        'to_stop': next_stop,
                        'from_lat': current_coords.iloc[0]['stop_lat'],
                        'from_lon': current_coords.iloc[0]['stop_lon'],
                        'to_lat': next_coords.iloc[0]['stop_lat'],
                        'to_lon': next_coords.iloc[0]['stop_lon'],
                        'sequence': i
                    }
                    route_graph['segments'].append(segment)

            return route_graph

        except Exception as e:
            self.logger.error(f"Failed to create route graph: {e}")
            return {}

    def calculate_scheduled_headways(self, route_id: str, 
                                   time_window: Tuple[str, str] = ("06:00:00", "22:00:00")) -> Dict:
        """Calculate scheduled headways for a route"""
        try:
            # Get trips for the route
            route_trips = self.trips[self.trips['route_id'] == route_id]

            # Get departure times from first stop
            first_stop_times = []

            for trip_id in route_trips['trip_id']:
                trip_stops = self.stop_times[self.stop_times['trip_id'] == trip_id]
                if not trip_stops.empty:
                    first_stop = trip_stops.sort_values('stop_sequence').iloc[0]
                    departure_time = first_stop['departure_time']
                    first_stop_times.append(departure_time)

            # Convert times to seconds for calculation
            def time_to_seconds(time_str):
                try:
                    h, m, s = map(int, time_str.split(':'))
                    return h * 3600 + m * 60 + s
                except:
                    return 0

            departure_seconds = [time_to_seconds(t) for t in first_stop_times]
            departure_seconds.sort()

            # Calculate headways
            headways = []
            for i in range(1, len(departure_seconds)):
                headway = departure_seconds[i] - departure_seconds[i-1]
                headways.append(headway)

            # Calculate statistics
            headway_stats = {
                'mean_headway': np.mean(headways) if headways else 0,
                'std_headway': np.std(headways) if headways else 0,
                'min_headway': np.min(headways) if headways else 0,
                'max_headway': np.max(headways) if headways else 0,
                'headway_count': len(headways)
            }

            return headway_stats

        except Exception as e:
            self.logger.error(f"Failed to calculate headways: {e}")
            return {}

    def get_route_list(self) -> List[Dict]:
        """Get list of available routes"""
        if self.routes is not None:
            return self.routes[['route_id', 'route_short_name', 
                              'route_long_name']].to_dict('records')
        return []

    def save_to_database(self) -> bool:
        """Save processed GTFS data to database"""
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))

            # Save each DataFrame to database
            if self.stops is not None:
                self.stops.to_sql('stops', conn, if_exists='replace', index=False)
            if self.routes is not None:
                self.routes.to_sql('routes', conn, if_exists='replace', index=False)
            if self.trips is not None:
                self.trips.to_sql('trips', conn, if_exists='replace', index=False)
            if self.stop_times is not None:
                self.stop_times.to_sql('stop_times', conn, if_exists='replace', index=False)

            conn.close()
            self.logger.info("GTFS data saved to database")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save to database: {e}")
            return False

# Example usage
if __name__ == "__main__":
    processor = GTFSProcessor()

    print("GTFS Processor initialized")
    print("Key methods:")
    print("- download_gtfs_data()")
    print("- load_gtfs_files()")
    print("- create_route_graph()")
    print("- calculate_scheduled_headways()")
    # Example: check data source after loading files
    processor.load_gtfs_files("gtfs_data/")
    print(f"Data source: {processor.data_source}")