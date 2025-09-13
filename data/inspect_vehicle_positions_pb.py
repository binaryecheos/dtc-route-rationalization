

import sys
import os
from google.transit import gtfs_realtime_pb2
import pandas as pd

PB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VehiclePositions.pb"))

def parse_vehicle_positions_pb(pb_path):
    feed = gtfs_realtime_pb2.FeedMessage()
    with open(pb_path, "rb") as f:
        feed.ParseFromString(f.read())
    data = []
    for entity in feed.entity:
        if entity.HasField('vehicle'):
            vehicle = entity.vehicle
            entry = {
                'vehicle_id': vehicle.vehicle.id if vehicle.HasField('vehicle') else None,
                'trip_id': vehicle.trip.trip_id if vehicle.HasField('trip') else None,
                'route_id': vehicle.trip.route_id if vehicle.HasField('trip') else None,
                'latitude': vehicle.position.latitude if vehicle.HasField('position') else None,
                'longitude': vehicle.position.longitude if vehicle.HasField('position') else None,
                'speed': vehicle.position.speed if vehicle.HasField('position') and vehicle.position.HasField('speed') else None,
                'bearing': vehicle.position.bearing if vehicle.HasField('position') and vehicle.position.HasField('bearing') else None,
                'timestamp': vehicle.timestamp if vehicle.HasField('timestamp') else None
            }
            data.append(entry)
    df = pd.DataFrame(data)
    return df


import sqlite3

def create_or_update_vehicle_positions_table(db_path, df):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT,
            trip_id TEXT,
            route_id TEXT,
            latitude REAL,
            longitude REAL,
    """
    This script is now deprecated.
    All real-time vehicle position parsing, DataFrame creation, and database updates are handled in data/realtime_archiver.py.
    Please use RealtimeArchiver for all real-time data processing.
    """
            speed REAL,
            bearing REAL,
            timestamp INTEGER
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_id ON vehicle_positions(vehicle_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trip_id ON vehicle_positions(trip_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_route_id ON vehicle_positions(route_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_positions(timestamp)')
    conn.commit()

    # Optional: Clear old data before inserting new
    cursor.execute('DELETE FROM vehicle_positions')
    conn.commit()

    # Insert new data
    df.to_sql('vehicle_positions', conn, if_exists='append', index=False)
    conn.close()

if __name__ == "__main__":
    df = parse_vehicle_positions_pb(PB_PATH)
    print("Optimized DataFrame loaded with shape:", df.shape)
    print(df.head())
    DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bus_data.db"))
    create_or_update_vehicle_positions_table(DB_PATH, df)
    print(f"Data inserted into database table 'vehicle_positions'.")
