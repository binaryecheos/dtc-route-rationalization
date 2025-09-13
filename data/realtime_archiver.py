import requests
import sqlite3
import pandas as pd
from google.transit import gtfs_realtime_pb2
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import threading
from dataclasses import dataclass
import json
import http.client  # <--- For IncompleteRead error

@dataclass
class VehiclePosition:
    """Data class for vehicle position records"""
    timestamp: int
    vehicle_id: str
    trip_id: str
    route_id: str
    latitude: float
    longitude: float
    speed: Optional[float] = None
    bearing: Optional[float] = None

class RealtimeArchiver:
    """
    Archives and provides real-time GTFS-RT vehicle position data from Delhi OTD portal.
    Use get_vehicle_positions_df() to retrieve latest positions as a pandas DataFrame for downstream modules.
    """
    def __init__(self, api_key: str, database_url: str = "sqlite:///bus_data.db", archive_frequency: int = 30):
        self.api_key = api_key
        self.database_url = database_url
        self.archive_frequency = archive_frequency
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.archive_thread = None
        # Initialize database
        self._init_database()

    def _init_database(self):
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))
            conn.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    trip_id TEXT,
                    route_id TEXT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    speed REAL,
                    bearing REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON vehicle_positions(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicle_route 
                ON vehicle_positions(vehicle_id, route_id)
            ''')
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def fetch_realtime_data(self, max_retries=3, backoff_factor=2) -> List[VehiclePosition]:
        """Fetch real-time vehicle positions from Delhi OTD API with retry on incomplete read/network fail"""
        url = f"https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb"
        params = {'key': self.api_key}
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(response.content)

                positions = []
                data = []
                for entity in feed.entity:
                    if entity.HasField('vehicle'):
                        vehicle = entity.vehicle
                        if vehicle.HasField('position'):
                            pos_bearing = getattr(vehicle.position, 'bearing', None)
                            pos_speed = getattr(vehicle.position, 'speed', None)
                            vehicle_id = getattr(vehicle.vehicle, 'id', '') if vehicle.HasField('vehicle') else ''
                            trip_id = getattr(vehicle.trip, 'trip_id', '') if vehicle.HasField('trip') else ''
                            route_id = getattr(vehicle.trip, 'route_id', '') if vehicle.HasField('trip') else ''
                            position = VehiclePosition(
                                timestamp=feed.header.timestamp,
                                vehicle_id=vehicle_id,
                                trip_id=trip_id,
                                route_id=route_id,
                                latitude=vehicle.position.latitude,
                                longitude=vehicle.position.longitude,
                                speed=pos_speed,
                                bearing=pos_bearing
                            )
                            positions.append(position)
                            data.append({
                                'timestamp': feed.header.timestamp,
                                'vehicle_id': vehicle_id,
                                'trip_id': trip_id,
                                'route_id': route_id,
                                'latitude': vehicle.position.latitude,
                                'longitude': vehicle.position.longitude,
                                'speed': pos_speed,
                                'bearing': pos_bearing
                            })
                self._latest_df = pd.DataFrame(data)
                self.logger.info(f"Fetched {len(positions)} vehicle positions")
                return positions

            except (requests.exceptions.RequestException, http.client.IncompleteRead) as e:
                self.logger.error(f"Network error fetching real-time data (attempt {attempt+1}): {e}")
                self._latest_df = pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error parsing real-time data (attempt {attempt+1}): {e}")
                self._latest_df = pd.DataFrame()
            attempt += 1
            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        self.logger.error("All attempts to fetch real-time data failed.")
        self._latest_df = pd.DataFrame()
        return []

    def get_vehicle_positions_df(self) -> pd.DataFrame:
        """
        Returns the latest vehicle positions as a pandas DataFrame.
        Columns: timestamp, vehicle_id, trip_id, route_id, latitude, longitude, speed, bearing
        """
        if hasattr(self, '_latest_df'):
            return self._latest_df.copy()
        else:
            self.fetch_realtime_data()
            return getattr(self, '_latest_df', pd.DataFrame()).copy()

    def save_positions(self, positions: List[VehiclePosition]) -> bool:
        """Save vehicle positions to database"""
        if not positions:
            return True
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))
            data = [(
                pos.timestamp,
                pos.vehicle_id,
                pos.trip_id,
                pos.route_id,
                pos.latitude,
                pos.longitude,
                pos.speed,
                pos.bearing
            ) for pos in positions]
            conn.executemany('''
                INSERT INTO vehicle_positions 
                (timestamp, vehicle_id, trip_id, route_id, latitude, longitude, speed, bearing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
            conn.close()
            self.logger.info(f"Saved {len(positions)} positions to database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save positions: {e}")
            return False

    def calculate_headways(self, route_id: str, stop_id: str, time_window: int = 3600) -> List[float]:
        """Calculate actual headways from archived data"""
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))
            end_time = int(time.time())
            start_time = end_time - time_window
            query = '''
                SELECT timestamp, vehicle_id, latitude, longitude
                FROM vehicle_positions 
                WHERE route_id = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            df = pd.read_sql_query(query, conn, params=[route_id, start_time, end_time])
            conn.close()
            if df.empty:
                return []
            headways = []
            vehicles = df['vehicle_id'].unique()
            for vehicle in vehicles:
                vehicle_data = df[df['vehicle_id'] == vehicle].sort_values('timestamp')
                if len(vehicle_data) > 1:
                    time_diffs = vehicle_data['timestamp'].diff().dropna()
                    headways.extend(time_diffs.tolist())
            return headways
        except Exception as e:
            self.logger.error(f"Failed to calculate headways: {e}")
            return []

    def get_route_performance(self, route_id: str, hours: int = 24) -> Dict:
        """Get performance metrics for a route"""
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))
            end_time = int(time.time())
            start_time = end_time - (hours * 3600)
            query = '''
                SELECT COUNT(*) as position_count,
                       COUNT(DISTINCT vehicle_id) as vehicle_count,
                       MIN(timestamp) as first_record,
                       MAX(timestamp) as last_record
                FROM vehicle_positions 
                WHERE route_id = ? 
                AND timestamp BETWEEN ? AND ?
            '''
            result = pd.read_sql_query(query, conn, params=[route_id, start_time, end_time])
            conn.close()
            if not result.empty:
                return result.iloc.to_dict()
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get route performance: {e}")
            return {}

    def start_archiving(self):
        """Start the real-time data archiving process"""
        if self.is_running:
            self.logger.warning("Archiving already running")
            return
        self.is_running = True
        self.archive_thread = threading.Thread(target=self._archive_loop)
        self.archive_thread.daemon = True
        self.archive_thread.start()
        self.logger.info("Real-time data archiving started")

    def stop_archiving(self):
        """Stop the real-time data archiving process"""
        self.is_running = False
        if self.archive_thread:
            self.archive_thread.join(timeout=5)
        self.logger.info("Real-time data archiving stopped")

    def _archive_loop(self):
        """Main archiving loop"""
        while self.is_running:
            try:
                positions = self.fetch_realtime_data()
                if positions:
                    self.save_positions(positions)
                time.sleep(self.archive_frequency)
            except Exception as e:
                self.logger.error(f"Error in archive loop: {e}")
                time.sleep(self.archive_frequency)

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old archived data"""
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))
            cutoff_time = int(time.time()) - (days_to_keep * 24 * 3600)
            cursor = conn.execute(
                'DELETE FROM vehicle_positions WHERE timestamp < ?', 
                (cutoff_time,)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            self.logger.info(f"Cleaned up {deleted_count} old records")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")

# Example usage with a loop for multiple archivers
if __name__ == "__main__":
    api_key = "L5jVvGl6iLEdSqnZG42pEm5LD94t1PYF"  # Single API key
    archiver = RealtimeArchiver(api_key=api_key, archive_frequency=120)  # archive_frequency is not used here

    n = 10  # Number of cycles; change as per your requirement
    for i in range(n):
        print(f"Iteration {i+1}: Fetching real-time vehicle positions...")
        positions = archiver.fetch_realtime_data()
        if positions:
            print(f"Fetched {len(positions)} vehicle positions.")
            archiver.save_positions(positions)
            print("Positions saved to database.")
        else:
            print("No vehicle positions fetched or an error occurred.")
        if i < n - 1:  # Do not sleep after last iteration
            print(f"Waiting 120 seconds before next fetch...")
            time.sleep(120)
    print("Completed all iterations.")
