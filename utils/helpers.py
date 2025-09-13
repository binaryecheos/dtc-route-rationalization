import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from geopy.distance import geodesic
import sqlite3

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters
    r = 6371000

    return r * c

def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS time string to seconds since midnight"""

    try:
        if ':' in time_str:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds = int(parts[2]) if len(parts) > 2 else 0

            return hours * 3600 + minutes * 60 + seconds
        else:
            return int(time_str)
    except (ValueError, IndexError):
        return 0

def seconds_to_time(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM:SS string"""

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def interpolate_position(start_pos: Tuple[float, float], 
                        end_pos: Tuple[float, float], 
                        progress: float) -> Tuple[float, float]:
    """
    Interpolate position between two geographic points
    progress should be between 0 and 1
    """

    lat1, lon1 = start_pos
    lat2, lon2 = end_pos

    # Simple linear interpolation (works for short distances)
    lat = lat1 + progress * (lat2 - lat1)
    lon = lon1 + progress * (lon2 - lon1)

    return (lat, lon)

def calculate_headway_from_timestamps(timestamps: List[int]) -> List[float]:
    """Calculate headways from a list of timestamps"""

    if len(timestamps) < 2:
        return []

    sorted_timestamps = sorted(timestamps)
    headways = []

    for i in range(1, len(sorted_timestamps)):
        headway = sorted_timestamps[i] - sorted_timestamps[i-1]
        headways.append(headway)

    return headways

def smooth_time_series(data: List[float], window_size: int = 5) -> List[float]:
    """Apply moving average smoothing to time series data"""

    if len(data) < window_size:
        return data

    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)

        window_data = data[start_idx:end_idx]
        smoothed.append(sum(window_data) / len(window_data))

    return smoothed

def detect_outliers(data: List[float], method: str = 'iqr', 
                   threshold: float = 1.5) -> List[bool]:
    """
    Detect outliers in data using IQR or z-score method
    Returns boolean mask where True indicates outlier
    """

    if not data:
        return []

    data_array = np.array(data)

    if method == 'iqr':
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (data_array < lower_bound) | (data_array > upper_bound)

    elif method == 'zscore':
        mean = np.mean(data_array)
        std = np.std(data_array)

        if std == 0:
            return [False] * len(data)

        z_scores = np.abs((data_array - mean) / std)
        outliers = z_scores > threshold

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

    return outliers.tolist()

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def validate_gtfs_data(gtfs_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    Validate GTFS data for completeness and consistency
    Returns dictionary of validation issues
    """

    issues = {}

    required_files = ['stops', 'routes', 'trips', 'stop_times']

    for file_name in required_files:
        if file_name not in gtfs_data:
            issues[file_name] = [f"Required file {file_name}.txt is missing"]
            continue

        df = gtfs_data[file_name]
        file_issues = []

        if df.empty:
            file_issues.append(f"{file_name}.txt is empty")

        # File-specific validations
        if file_name == 'stops':
            required_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                file_issues.append(f"Missing columns: {missing_cols}")

            # Check for valid coordinates
            if 'stop_lat' in df.columns and 'stop_lon' in df.columns:
                invalid_coords = df[(df['stop_lat'].abs() > 90) | 
                                  (df['stop_lon'].abs() > 180)]
                if not invalid_coords.empty:
                    file_issues.append(f"{len(invalid_coords)} stops have invalid coordinates")

        elif file_name == 'routes':
            required_cols = ['route_id', 'route_short_name', 'route_long_name']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                file_issues.append(f"Missing columns: {missing_cols}")

        elif file_name == 'trips':
            required_cols = ['route_id', 'service_id', 'trip_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                file_issues.append(f"Missing columns: {missing_cols}")

        elif file_name == 'stop_times':
            required_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                file_issues.append(f"Missing columns: {missing_cols}")

        if file_issues:
            issues[file_name] = file_issues

    return issues

def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return {}

def save_results_to_json(results: Dict[str, Any], filepath: str):
    """Save results dictionary to JSON file"""

    # Convert datetime objects to strings for JSON serialization
    def serialize_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=serialize_datetime)
        logging.info(f"Results saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save results to {filepath}: {e}")

def create_database_backup(database_url: str, backup_path: str):
    """Create a backup of the SQLite database"""

    try:
        source_db = sqlite3.connect(database_url.replace("sqlite:///", ""))
        backup_db = sqlite3.connect(backup_path)

        source_db.backup(backup_db)

        source_db.close()
        backup_db.close()

        logging.info(f"Database backup created: {backup_path}")

    except Exception as e:
        logging.error(f"Failed to create database backup: {e}")

def setup_project_logging(log_level: str = "INFO", log_file: str = "bus_system.log"):
    """Setup logging configuration for the project"""

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Suppress some verbose libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def calculate_service_metrics(headways: List[float], target_headway: float = 600) -> Dict[str, float]:
    """Calculate comprehensive service quality metrics from headway data"""

    if not headways:
        return {}

    headways_array = np.array(headways)

    metrics = {
        # Basic statistics
        'mean_headway': np.mean(headways_array),
        'median_headway': np.median(headways_array),
        'std_headway': np.std(headways_array),
        'min_headway': np.min(headways_array),
        'max_headway': np.max(headways_array),

        # Regularity metrics
        'cv_headway': np.std(headways_array) / np.mean(headways_array) if np.mean(headways_array) > 0 else 0,
        'headway_adherence': 1 - abs(np.mean(headways_array) - target_headway) / target_headway,

        # Service quality indicators
        'bunching_frequency': sum(1 for h in headways if h < target_headway * 0.5) / len(headways),
        'large_gap_frequency': sum(1 for h in headways if h > target_headway * 1.5) / len(headways),

        # Passenger impact (estimated)
        'estimated_wait_time': sum(h**2 for h in headways) / (2 * sum(headways)),
        'wait_time_reliability': 1 / (1 + np.var(headways_array) / (np.mean(headways_array)**2))
    }

    return metrics

class DataQualityChecker:
    """Utility class for checking data quality in transit datasets"""

    @staticmethod
    def check_coordinate_validity(lat: float, lon: float) -> bool:
        """Check if coordinates are valid"""
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def check_time_format(time_str: str) -> bool:
        """Check if time string is in valid HH:MM:SS format"""
        try:
            parts = time_str.split(':')
            if len(parts) != 3:
                return False

            hours, minutes, seconds = map(int, parts)
            return 0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def check_headway_reasonableness(headway: float, min_headway: float = 60, 
                                   max_headway: float = 3600) -> bool:
        """Check if headway is within reasonable bounds"""
        return min_headway <= headway <= max_headway

    @staticmethod
    def identify_data_gaps(timestamps: List[int], max_gap: int = 300) -> List[Tuple[int, int]]:
        """Identify gaps in timestamp data larger than max_gap seconds"""

        if len(timestamps) < 2:
            return []

        sorted_timestamps = sorted(timestamps)
        gaps = []

        for i in range(1, len(sorted_timestamps)):
            gap_size = sorted_timestamps[i] - sorted_timestamps[i-1]
            if gap_size > max_gap:
                gaps.append((sorted_timestamps[i-1], sorted_timestamps[i]))

        return gaps
