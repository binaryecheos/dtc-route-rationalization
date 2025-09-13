import requests
import logging
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import json

@dataclass
class TrafficPrediction:
    """Data class for traffic predictions"""
    origin: Tuple[float, float]
    destination: Tuple[float, float] 
    duration_seconds: int
    distance_meters: int
    traffic_duration_seconds: int
    departure_time: str
    status: str

class GoogleMapsTrafficAPI:
    """
    Integrates with Google Maps Routes API to provide real-time traffic-aware
    travel time predictions for bus route segments.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://maps.googleapis.com/maps/api"

        # Rate limiting
        self.requests_per_minute = 100  # Adjust based on API quota
        self.last_request_time = 0

    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute

        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()

    def get_travel_time(self, origin: Tuple[float, float], 
                       destination: Tuple[float, float],
                       departure_time: str = "now") -> Optional[TrafficPrediction]:
        """
        Get travel time between two points using Google Maps API

        Args:
            origin: (latitude, longitude) tuple
            destination: (latitude, longitude) tuple  
            departure_time: "now" or timestamp
        """
        try:
            self._rate_limit()

            # Format coordinates
            origin_str = f"{origin[0]},{origin[1]}"
            dest_str = f"{destination[0]},{destination[1]}"

            # Prepare API request
            url = f"{self.base_url}/directions/json"
            params = {
                'origin': origin_str,
                'destination': dest_str,
                'departure_time': departure_time,
                'traffic_model': 'best_guess',
                'key': self.api_key,
                'mode': 'driving'  # Use driving for bus routes
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 'OK':
                self.logger.error(f"API error: {data.get('error_message', data['status'])}")
                return None

            if not data['routes']:
                self.logger.warning("No routes found")
                return None

            # Extract first route information
            route = data['routes'][0]
            leg = route['legs'][0]

            # Get duration in traffic if available
            traffic_duration = leg.get('duration_in_traffic', leg['duration'])

            prediction = TrafficPrediction(
                origin=origin,
                destination=destination,
                duration_seconds=leg['duration']['value'],
                distance_meters=leg['distance']['value'],
                traffic_duration_seconds=traffic_duration['value'],
                departure_time=departure_time,
                status='success'
            )

            return prediction

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting travel time: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting travel time: {e}")
            return None

    def get_route_segment_times(self, route_segments: List[Dict]) -> Dict[str, TrafficPrediction]:
        """
        Get travel times for multiple route segments

        Args:
            route_segments: List of segment dictionaries with 'from_lat', 'from_lon', 'to_lat', 'to_lon'
        """
        segment_times = {}

        for i, segment in enumerate(route_segments):
            try:
                origin = (segment['from_lat'], segment['from_lon'])
                destination = (segment['to_lat'], segment['to_lon'])

                prediction = self.get_travel_time(origin, destination)

                if prediction:
                    segment_key = f"segment_{i}"
                    segment_times[segment_key] = prediction

                    self.logger.info(f"Segment {i}: {prediction.traffic_duration_seconds}s")

                # Small delay between requests to avoid hitting rate limits
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing segment {i}: {e}")
                continue

        return segment_times

    def get_batch_travel_times(self, origin_dest_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> List[TrafficPrediction]:
        """
        Get travel times for multiple origin-destination pairs
        """
        predictions = []

        for origin, destination in origin_dest_pairs:
            prediction = self.get_travel_time(origin, destination)
            if prediction:
                predictions.append(prediction)

        return predictions

    def calculate_segment_baseline(self, route_segments: List[Dict], 
                                 samples: int = 10) -> Dict[str, Dict]:
        """
        Calculate baseline travel times for route segments by sampling over time
        """
        segment_baselines = {}

        for i, segment in enumerate(route_segments):
            segment_key = f"segment_{i}"
            times = []

            # Collect multiple samples
            for _ in range(samples):
                origin = (segment['from_lat'], segment['from_lon'])
                destination = (segment['to_lat'], segment['to_lon'])

                prediction = self.get_travel_time(origin, destination)

                if prediction:
                    times.append(prediction.traffic_duration_seconds)

                time.sleep(1)  # Wait between samples

            if times:
                baseline = {
                    'mean_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'samples': len(times),
                    'segment_info': segment
                }
                segment_baselines[segment_key] = baseline

        return segment_baselines

    def estimate_arrival_time(self, current_position: Tuple[float, float],
                            upcoming_stops: List[Tuple[float, float]]) -> List[int]:
        """
        Estimate arrival times at upcoming stops
        """
        arrival_times = []
        current_pos = current_position
        current_time = 0

        for stop_position in upcoming_stops:
            prediction = self.get_travel_time(current_pos, stop_position)

            if prediction:
                current_time += prediction.traffic_duration_seconds
                arrival_times.append(current_time)
                current_pos = stop_position
            else:
                # Fallback: estimate based on average speed
                # This is simplified - would use more sophisticated fallback in production
                arrival_times.append(current_time + 300)  # 5 minute estimate
                current_time += 300

        return arrival_times

    def get_traffic_conditions(self, center: Tuple[float, float], 
                             radius_km: float = 5) -> Dict:
        """
        Get general traffic conditions in an area (simplified implementation)
        """
        try:
            # This would require additional APIs or different approach
            # For now, return a mock response structure

            conditions = {
                'center': center,
                'radius_km': radius_km,
                'traffic_level': 'moderate',  # light, moderate, heavy
                'average_speed_kmh': 25,
                'incidents': [],
                'timestamp': int(time.time())
            }

            return conditions

        except Exception as e:
            self.logger.error(f"Error getting traffic conditions: {e}")
            return {}

    def validate_api_key(self) -> bool:
        """Validate API key by making a simple request"""
        try:
            # Simple test request
            test_origin = (28.6139, 77.2090)  # Delhi center
            test_dest = (28.6200, 77.2200)    # Nearby point

            prediction = self.get_travel_time(test_origin, test_dest)
            return prediction is not None

        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

class MockTrafficAPI(GoogleMapsTrafficAPI):
    """
    Mock implementation for testing when API key is not available
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_travel_time(self, origin: Tuple[float, float], 
                       destination: Tuple[float, float],
                       departure_time: str = "now") -> Optional[TrafficPrediction]:
        """Mock implementation returning estimated travel times"""

        # Simple distance calculation (Haversine formula would be more accurate)
        lat1, lon1 = origin
        lat2, lon2 = destination

        # Rough distance calculation in kilometers
        distance_km = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111

        # Estimate travel time based on average urban speed (25 km/h)
        travel_time_hours = distance_km / 25
        travel_time_seconds = int(travel_time_hours * 3600)

        # Add some traffic variability
        import random
        traffic_multiplier = random.uniform(1.0, 1.5)
        traffic_time_seconds = int(travel_time_seconds * traffic_multiplier)

        prediction = TrafficPrediction(
            origin=origin,
            destination=destination,
            duration_seconds=travel_time_seconds,
            distance_meters=int(distance_km * 1000),
            traffic_duration_seconds=traffic_time_seconds,
            departure_time=departure_time,
            status='mock'
        )

        return prediction

# Example usage
if __name__ == "__main__":
    # For demonstration, use mock API
    traffic_api = MockTrafficAPI()

    # Test coordinates (Delhi area)
    origin = (28.6139, 77.2090)
    destination = (28.6200, 77.2200)

    prediction = traffic_api.get_travel_time(origin, destination)
    print(f"Travel prediction: {prediction}")

    print("Traffic API initialized")
    print("Key methods:")
    print("- get_travel_time()")
    print("- get_route_segment_times()")
    print("- estimate_arrival_time()")