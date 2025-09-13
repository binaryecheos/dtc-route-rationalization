import simpy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import random

@dataclass
class SimulationBus:
    """Bus entity in the simulation"""
    bus_id: str
    route_id: str
    current_stop_idx: int = 0
    passenger_count: int = 0
    last_departure_time: float = 0
    total_delay: float = 0
    is_held: bool = False

@dataclass
class SimulationStop:
    """Bus stop entity in the simulation"""
    stop_id: str
    name: str
    lat: float
    lon: float
    waiting_passengers: int = 0
    passenger_arrival_rate: float = 0.1  # passengers per second

@dataclass
class SimulationEvent:
    """Base class for simulation events"""
    timestamp: float
    event_type: str
    bus_id: str
    stop_id: str
    data: Dict[str, Any] = field(default_factory=dict)

class EventDrivenTransitSimulator:
    def _bus_operation_process(self, bus_id: str) -> Generator:
        """Simulate bus movement along the route."""
        bus = self.buses[bus_id]
        stops_list = list(self.stops.keys())
        while True:
            # Arrive at current stop
            current_stop_id = stops_list[bus.current_stop_idx]
            yield self.env.process(self._bus_arrives_at_stop(bus_id, current_stop_id))

            # Make control decision (holding)
            holding_time = 0
            if self.control_policy:
                holding_time = self.control_policy(self._get_system_state(bus_id, current_stop_id))
            if holding_time > 0:
                bus.is_held = True
                bus.total_delay += holding_time
                yield self.env.timeout(holding_time)
                bus.is_held = False

            # Depart from stop
            yield self.env.process(self._bus_departs_from_stop(bus_id, current_stop_id))

            # Travel to next stop
            travel_time = self._get_segment_travel_time(bus.current_stop_idx)
            yield self.env.timeout(travel_time)

            # Move to next stop
            bus.current_stop_idx = (bus.current_stop_idx + 1) % len(stops_list)
    def _passenger_arrival_process(self, stop_id: str) -> Generator:
        """Simulate passenger arrivals at a stop."""
        stop = self.stops[stop_id]
        while True:
            # Increment waiting passengers based on arrival rate
            stop.waiting_passengers += 1
            # Wait for next arrival (inverse of arrival rate)
            yield self.env.timeout(1.0 / max(stop.passenger_arrival_rate, 0.01))
    """
    Event-driven simulation of bus transit system.
    Serves as the digital twin for RL agent training and system analysis.
    """

    def __init__(self, route_config: Dict[str, Any], 
                 simulation_config: Dict[str, Any]):

        self.route_config = route_config
        self.simulation_config = simulation_config
        self.logger = logging.getLogger(__name__)

        # Simulation environment
        self.env = None

        # Simulation entities
        self.buses = {}
        self.stops = {}
        self.route_segments = []

        # Event tracking
        self.events = []
        self.performance_metrics = {}

        # Control policies (can be injected)
        self.control_policy = None

        # Initialize simulation components
        self._initialize_network()

    def _initialize_network(self):
        """Initialize network topology from route configuration"""

        # Initialize stops
        for stop_data in self.route_config.get('stops', []):
            stop = SimulationStop(
                stop_id=stop_data['id'],
                name=stop_data.get('name', ''),
                lat=stop_data['lat'],
                lon=stop_data['lon'],
                passenger_arrival_rate=stop_data.get('arrival_rate', 0.1)
            )
            self.stops[stop.stop_id] = stop

        # Initialize route segments
        stops_list = list(self.stops.keys())
        for i in range(len(stops_list)):
            next_stop = stops_list[(i + 1) % len(stops_list)]

            segment = {
                'from_stop': stops_list[i],
                'to_stop': next_stop,
                'distance_m': self._calculate_distance(stops_list[i], next_stop),
                'base_travel_time': self._estimate_travel_time(stops_list[i], next_stop)
            }
            self.route_segments.append(segment)

        # Initialize buses
        num_buses = self.route_config.get('num_buses', 5)
        headway = self.simulation_config.get('target_headway', 600)

        for i in range(num_buses):
            bus = SimulationBus(
                bus_id=f"bus_{i}",
                route_id=self.route_config.get('route_id', 'route_1'),
                current_stop_idx=0,  # All start at first stop
                last_departure_time=i * headway  # Stagger initial departures
            )
            self.buses[bus.bus_id] = bus

    def _calculate_distance(self, stop1_id: str, stop2_id: str) -> float:
        """Calculate distance between two stops"""

        stop1 = self.stops[stop1_id]
        stop2 = self.stops[stop2_id]

        # Haversine formula (simplified)
        lat_diff = abs(stop2.lat - stop1.lat)
        lon_diff = abs(stop2.lon - stop1.lon)

        # Rough distance in meters
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111000

        return max(distance, 500)  # Minimum 500m between stops

    def _estimate_travel_time(self, stop1_id: str, stop2_id: str) -> float:
        """Estimate base travel time between stops"""

        distance = self._calculate_distance(stop1_id, stop2_id)

        # Base speed: 25 km/h in urban conditions
        base_speed_ms = 25 * 1000 / 3600  # Convert to m/s
        base_time = distance / base_speed_ms

        # Add stop dwell time
        dwell_time = 30  # 30 seconds average dwell

        return base_time + dwell_time

    def set_control_policy(self, policy_function):
        """Set control policy for making holding decisions"""
        self.control_policy = policy_function

    def run_simulation(self, duration: float) -> Dict[str, Any]:
        """Run the simulation for specified duration"""

        self.env = simpy.Environment()
        self.events = []

        # Start passenger arrival processes
        for stop_id in self.stops:
            self.env.process(self._passenger_arrival_process(stop_id))

        # Start bus processes
        for bus_id in self.buses:
            self.env.process(self._bus_operation_process(bus_id))

        # Run simulation
        self.logger.info(f"Starting simulation for {duration} seconds")
        self.env.run(until=duration)

        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()

        self.logger.info(f"Simulation completed. {len(self.events)} events recorded")
        return self.performance_metrics

    def _bus_arrives_at_stop(self, bus_id: str, stop_id: str) -> Generator:
        """Process bus arrival at stop"""

        bus = self.buses[bus_id]
        stop = self.stops[stop_id]

        # Record arrival event
        event = SimulationEvent(
            timestamp=self.env.now,
            event_type="bus_arrival",
            bus_id=bus_id,
            stop_id=stop_id,
            data={
                'passenger_count': bus.passenger_count,
                'waiting_passengers': stop.waiting_passengers,
                'delay': self.env.now - bus.last_departure_time
            }
        )
        self.events.append(event)

        # Passenger boarding/alighting
        yield self.env.process(self._passenger_exchange(bus_id, stop_id))

        yield self.env.timeout(0)  # Minimal processing time

    def _bus_departs_from_stop(self, bus_id: str, stop_id: str) -> Generator:
        """Process bus departure from stop"""

        bus = self.buses[bus_id]
        bus.last_departure_time = self.env.now

        # Record departure event
        event = SimulationEvent(
            timestamp=self.env.now,
            event_type="bus_departure", 
            bus_id=bus_id,
            stop_id=stop_id,
            data={
                'passenger_count': bus.passenger_count,
                'total_delay': bus.total_delay
            }
        )
        self.events.append(event)

        yield self.env.timeout(0)  # Minimal processing time

    def _passenger_exchange(self, bus_id: str, stop_id: str) -> Generator:
        """Process passenger boarding and alighting"""

        bus = self.buses[bus_id]
        stop = self.stops[stop_id]

        # Passengers alighting (random number based on load)
        if bus.passenger_count > 0:
            alighting_rate = 0.3  # 30% of passengers alight on average
            passengers_alighting = np.random.binomial(bus.passenger_count, alighting_rate)
            bus.passenger_count -= passengers_alighting

        # Passengers boarding (all waiting passengers board, subject to capacity)
        max_capacity = 60  # Typical bus capacity
        available_capacity = max_capacity - bus.passenger_count
        passengers_boarding = min(stop.waiting_passengers, available_capacity)

        bus.passenger_count += passengers_boarding
        stop.waiting_passengers -= passengers_boarding

        # Dwell time based on passenger exchange
        base_dwell = 10  # 10 seconds base
        passenger_dwell = (passengers_boarding + (passengers_alighting if 'passengers_alighting' in locals() else 0)) * 2
        total_dwell = base_dwell + passenger_dwell

        yield self.env.timeout(total_dwell)

    def _get_segment_travel_time(self, segment_idx: int) -> float:
        """Get travel time for route segment with variability"""

        if segment_idx >= len(self.route_segments):
            segment_idx = 0

        segment = self.route_segments[segment_idx]
        base_time = segment['base_travel_time']

        # Add traffic variability
        traffic_multiplier = np.random.lognormal(0, 0.3)  # Log-normal for traffic variability
        traffic_multiplier = np.clip(traffic_multiplier, 0.5, 2.0)  # Reasonable bounds

        # Time-of-day effect
        hour = (self.env.now % 86400) / 3600
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            rush_multiplier = 1.5
        else:
            rush_multiplier = 1.0

        return base_time * traffic_multiplier * rush_multiplier

    def _make_control_decision(self, bus_id: str, stop_id: str) -> float:
        """Make control decision using policy or default logic"""

        if self.control_policy is None:
            return 0  # No holding by default

        # Get current system state
        state = self._get_system_state(bus_id, stop_id)

        # Apply control policy
        try:
            holding_time = self.control_policy(state)
            return max(0, min(holding_time, 120))  # Clip to reasonable range
        except Exception as e:
            self.logger.error(f"Control policy error: {e}")
            return 0

    def _get_system_state(self, bus_id: str, stop_id: str) -> Dict[str, Any]:
        """Get current system state for control decisions"""

        bus = self.buses[bus_id]
        stop = self.stops[stop_id]

        # Calculate headways
        headways = self._calculate_current_headways(bus_id)

        state = {
            'bus_id': bus_id,
            'stop_id': stop_id,
            'current_time': self.env.now,
            'passenger_count': bus.passenger_count,
            'waiting_passengers': stop.waiting_passengers,
            'headway_forward': headways.get('forward', 600),
            'headway_backward': headways.get('backward', 600),
            'total_delay': bus.total_delay,
            'stop_index': bus.current_stop_idx
        }

        return state

    def _calculate_current_headways(self, target_bus_id: str) -> Dict[str, float]:
        """Calculate headways for a specific bus"""

        target_bus = self.buses[target_bus_id]

        # Find buses at same stop or nearby
        same_route_buses = [(bid, bus) for bid, bus in self.buses.items() 
                           if bus.route_id == target_bus.route_id]

        # Sort by current position
        sorted_buses = sorted(same_route_buses, 
                            key=lambda x: (x[1].current_stop_idx, x[1].last_departure_time))

        # Find position of target bus
        target_idx = next(i for i, (bid, _) in enumerate(sorted_buses) if bid == target_bus_id)

        headways = {}

        # Forward headway (to next bus)
        if len(sorted_buses) > 1:
            next_idx = (target_idx + 1) % len(sorted_buses)
            next_bus = sorted_buses[next_idx][1]

            if next_bus.current_stop_idx == target_bus.current_stop_idx:
                headways['forward'] = abs(next_bus.last_departure_time - target_bus.last_departure_time)
            else:
                # Estimate based on position difference
                stop_diff = (next_bus.current_stop_idx - target_bus.current_stop_idx) % len(self.stops)
                headways['forward'] = stop_diff * 120  # 2 minutes per stop estimate

        # Backward headway (from previous bus)
        if len(sorted_buses) > 1:
            prev_idx = (target_idx - 1) % len(sorted_buses)
            prev_bus = sorted_buses[prev_idx][1]

            if prev_bus.current_stop_idx == target_bus.current_stop_idx:
                headways['backward'] = abs(target_bus.last_departure_time - prev_bus.last_departure_time)
            else:
                stop_diff = (target_bus.current_stop_idx - prev_bus.current_stop_idx) % len(self.stops)
                headways['backward'] = stop_diff * 120

        return headways

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate simulation performance metrics"""

        # Extract departure events
        departure_events = [e for e in self.events if e.event_type == "bus_departure"]

        if not departure_events:
            return {}

        # Calculate headways from departure events
        headways_by_stop = {}

        for event in departure_events:
            stop_id = event.stop_id
            if stop_id not in headways_by_stop:
                headways_by_stop[stop_id] = []
            headways_by_stop[stop_id].append(event.timestamp)

        # Calculate headway statistics
        all_headways = []

        for stop_id, timestamps in headways_by_stop.items():
            timestamps.sort()
            stop_headways = [timestamps[i] - timestamps[i-1] 
                           for i in range(1, len(timestamps))]
            all_headways.extend(stop_headways)

        if not all_headways:
            return {}

        # Performance metrics
        target_headway = self.simulation_config.get('target_headway', 600)

        metrics = {
            'mean_headway': np.mean(all_headways),
            'std_headway': np.std(all_headways),
            'cv_headway': np.std(all_headways) / np.mean(all_headways) if np.mean(all_headways) > 0 else 0,
            'target_headway': target_headway,
            'headway_adherence': 1 - abs(np.mean(all_headways) - target_headway) / target_headway,
            'bunching_events': sum(1 for h in all_headways if h < target_headway * 0.5),
            'total_events': len(self.events),
            'total_delays': sum(bus.total_delay for bus in self.buses.values()),
            'avg_passenger_wait': self._estimate_passenger_wait_time(all_headways)
        }

        return metrics

    def _estimate_passenger_wait_time(self, headways: List[float]) -> float:
        """Estimate average passenger wait time from headways"""

        # For random passenger arrivals: E[W] = E[H²] / (2 * E[H])
        mean_headway = np.mean(headways)
        mean_headway_squared = np.mean([h**2 for h in headways])

        if mean_headway > 0:
            return mean_headway_squared / (2 * mean_headway)
        return 0

    def get_events_dataframe(self) -> pd.DataFrame:
        """Convert events to pandas DataFrame for analysis"""

        if not self.events:
            return pd.DataFrame()

        events_data = []

        for event in self.events:
            row = {
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'bus_id': event.bus_id,
                'stop_id': event.stop_id
            }
            row.update(event.data)
            events_data.append(row)

        return pd.DataFrame(events_data)

    def export_results(self, filepath: str):
        """Export simulation results to file"""

        results = {
            'config': {
                'route_config': self.route_config,
                'simulation_config': self.simulation_config
            },
            'performance_metrics': self.performance_metrics,
            'events': [
                {
                    'timestamp': e.timestamp,
                    'event_type': e.event_type,
                    'bus_id': e.bus_id,
                    'stop_id': e.stop_id,
                    'data': e.data
                }
                for e in self.events
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results exported to {filepath}")

# Example usage
if __name__ == "__main__":

    # Define route configuration
    route_config = {
        'route_id': 'route_34',
        'num_buses': 3,
        'stops': [
            {'id': 'stop_1', 'name': 'Terminal A', 'lat': 28.60, 'lon': 77.20, 'arrival_rate': 0.05},
            {'id': 'stop_2', 'name': 'Market Square', 'lat': 28.61, 'lon': 77.21, 'arrival_rate': 0.08},
            {'id': 'stop_3', 'name': 'Hospital', 'lat': 28.62, 'lon': 77.22, 'arrival_rate': 0.06},
            {'id': 'stop_4', 'name': 'University', 'lat': 28.63, 'lon': 77.23, 'arrival_rate': 0.10},
            {'id': 'stop_5', 'name': 'Terminal B', 'lat': 28.64, 'lon': 77.24, 'arrival_rate': 0.05}
        ]
    }

    simulation_config = {
        'target_headway': 600,  # 10 minutes
        'max_holding_time': 120,
        'random_seed': 42
    }

    # Simple control policy example
    def simple_holding_policy(state):
        """Simple rule-based holding policy"""
        target_headway = 600
        forward_headway = state.get('headway_forward', target_headway)

        if forward_headway < target_headway * 0.7:  # Too close to next bus
            return 60  # Hold for 1 minute
        return 0  # No holding

    # Create and run simulation
    simulator = EventDrivenTransitSimulator(route_config, simulation_config)
    simulator.set_control_policy(simple_holding_policy)

    print("Running simulation...")
    results = simulator.run_simulation(duration=3600)  # 1 hour simulation

    print("\nSimulation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

    # Get events DataFrame
    events_df = simulator.get_events_dataframe()
    print(f"\nRecorded {len(events_df)} events")
    print(events_df['event_type'].value_counts())