import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

@dataclass
class BusAgent:
    """Represents a bus agent in the environment"""
    bus_id: str
    route_id: str
    current_stop_idx: int
    position: Tuple[float, float]  # lat, lon
    passenger_load: int
    last_departure_time: float
    speed: float

@dataclass
class BusEnvironmentState:
    """Complete state of the bus system"""
    buses: List[BusAgent]
    current_time: float
    headways: Dict[str, float]
    passenger_wait_times: Dict[str, List[float]]
    traffic_conditions: Dict[str, float]

class BusControlEnvironment(gym.Env):
    """
    OpenAI Gym environment for training RL agents on bus control.
    Implements the MDP formulation described in the research document.
    """

    def __init__(self, route_config: Dict[str, Any], 
                 simulation_params: Dict[str, Any]):
        super(BusControlEnvironment, self).__init__()

        self.route_config = route_config
        self.simulation_params = simulation_params
        self.logger = logging.getLogger(__name__)

        # Environment parameters
        self.target_headway = simulation_params.get('target_headway', 600)  # 10 minutes
        self.max_holding_time = simulation_params.get('max_holding_time', 120)  # 2 minutes
        self.simulation_time_step = simulation_params.get('time_step', 30)  # 30 seconds

        # State and action spaces
        self._setup_spaces()

        # Environment state
        self.current_state = None
        self.buses = []
        self.stops = route_config.get('stops', [])
        self.route_segments = route_config.get('segments', [])

        # Performance tracking
        self.episode_rewards = []
        self.episode_headway_variance = []
        self.episode_passenger_wait_times = []

        # Reset environment
        self.reset()

    def _setup_spaces(self):
        """Define observation and action spaces"""

        # Observation space: [stop_id, time_of_day, headway_fwd, headway_bwd, 
        #                    next_seg_time, passenger_load, traffic_speed]
        obs_low = np.array([
            0,      # stop_id (normalized)
            0,      # time_of_day (0-24 hours, normalized)
            0,      # headway_forward (seconds, normalized)
            0,      # headway_backward (seconds, normalized) 
            60,     # next_segment_travel_time (min 1 minute)
            0,      # passenger_load (normalized)
            5       # traffic_speed (min 5 km/h)
        ], dtype=np.float32)

        obs_high = np.array([
            1,      # stop_id (normalized)
            1,      # time_of_day (normalized)
            1,      # headway_forward (normalized by max_headway)
            1,      # headway_backward (normalized)
            1800,   # next_segment_travel_time (max 30 minutes)
            100,    # passenger_load (max capacity)
            60      # traffic_speed (max 60 km/h)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: Discrete actions for holding decisions
        # 0: Depart immediately, 1: Hold 30s, 2: Hold 60s, 3: Hold 90s
        self.action_space = spaces.Discrete(4)

        self.action_meanings = {
            0: "depart_now",
            1: "hold_30s", 
            2: "hold_60s",
            3: "hold_90s"
        }

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""

        # Initialize buses
        num_buses = self.route_config.get('num_buses', 5)
        self.buses = []

        for i in range(num_buses):
            bus = BusAgent(
                bus_id=f"bus_{i}",
                route_id=self.route_config.get('route_id', 'route_1'),
                current_stop_idx=i * len(self.stops) // num_buses,  # Distribute initially
                position=self._get_stop_position(i * len(self.stops) // num_buses),
                passenger_load=np.random.randint(10, 40),
                last_departure_time=0,
                speed=25  # km/h
            )
            self.buses.append(bus)

        # Initialize environment state
        self.current_time = 0
        self.episode_step = 0
        self.total_passenger_wait_time = 0
        self.total_control_delay = 0

        # Performance tracking
        self.episode_rewards = []
        self.episode_headway_variance = []

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step in the environment"""

        # Get current bus (for multi-agent, this would be more complex)
        current_bus = self._get_current_decision_bus()

        if current_bus is None:
            # No bus needs decision, advance time
            self.current_time += self.simulation_time_step
            return self._get_observation(), 0, False, {}

        # Apply action
        holding_time = self._apply_action(current_bus, action)

        # Advance simulation
        self._advance_simulation(holding_time)

        # Calculate reward
        reward = self._calculate_reward(current_bus, action, holding_time)

        # Check if episode is done
        done = self._is_episode_done()

        # Collect info
        info = {
            'bus_id': current_bus.bus_id,
            'action_taken': self.action_meanings[action],
            'holding_time': holding_time,
            'current_headway_variance': self._calculate_headway_variance(),
            'episode_step': self.episode_step
        }

        self.episode_step += 1
        self.episode_rewards.append(reward)

        return self._get_observation(), reward, done, info

    def _get_current_decision_bus(self) -> Optional[BusAgent]:
        """Get the bus that needs to make a decision"""

        # Simple policy: find bus that has been at stop longest
        # In practice, this would be based on actual arrival events

        decision_bus = None
        max_wait_time = 0

        for bus in self.buses:
            # Check if bus is at a stop and ready to depart
            stop_wait_time = self.current_time - bus.last_departure_time

            if stop_wait_time > max_wait_time and stop_wait_time > 60:  # At least 1 minute
                decision_bus = bus
                max_wait_time = stop_wait_time

        return decision_bus

    def _apply_action(self, bus: BusAgent, action: int) -> float:
        """Apply the chosen action to the bus"""

        holding_times = {0: 0, 1: 30, 2: 60, 3: 90}
        holding_time = holding_times[action]

        # Update bus departure time
        bus.last_departure_time = self.current_time + holding_time

        # Track total control delay
        self.total_control_delay += holding_time

        return holding_time

    def _advance_simulation(self, holding_time: float):
        """Advance the simulation by one time step"""

        # Advance time
        self.current_time += max(self.simulation_time_step, holding_time)

        # Update bus positions and states
        for bus in self.buses:
            # Simple movement model - buses advance along route
            if self.current_time >= bus.last_departure_time:
                # Bus is moving
                travel_time = self._get_segment_travel_time(bus.current_stop_idx)

                if self.current_time >= bus.last_departure_time + travel_time:
                    # Bus reaches next stop
                    bus.current_stop_idx = (bus.current_stop_idx + 1) % len(self.stops)
                    bus.position = self._get_stop_position(bus.current_stop_idx)

                    # Pick up passengers (affects load)
                    passengers_boarding = np.random.poisson(5)
                    passengers_alighting = min(bus.passenger_load, np.random.poisson(3))
                    bus.passenger_load = bus.passenger_load - passengers_alighting + passengers_boarding
                    bus.passenger_load = max(0, min(100, bus.passenger_load))  # Clip to capacity

        # Update passenger wait times
        self._update_passenger_wait_times()

    def _calculate_reward(self, bus: BusAgent, action: int, holding_time: float) -> float:
        """
        Calculate reward based on the MDP reward function described in the document:
        R = w1 * -headway_deviation² + w2 * -passenger_wait_time + w3 * -action_cost
        """

        # Calculate current headways
        headways = self._calculate_current_headways()

        # Weight parameters
        w1 = 1.0    # Headway regularity weight
        w2 = 0.5    # Passenger wait time weight  
        w3 = 0.1    # Action cost weight

        # Headway deviation penalty
        bus_headway_fwd = headways.get(f"{bus.bus_id}_forward", self.target_headway)
        bus_headway_bwd = headways.get(f"{bus.bus_id}_backward", self.target_headway)

        headway_deviation_fwd = (bus_headway_fwd - self.target_headway) ** 2
        headway_deviation_bwd = (bus_headway_bwd - self.target_headway) ** 2
        headway_penalty = w1 * (headway_deviation_fwd + headway_deviation_bwd) / (self.target_headway ** 2)

        # Passenger wait time penalty (estimated)
        estimated_wait_time = self._estimate_passenger_wait_time(bus)
        wait_time_penalty = w2 * estimated_wait_time / 600  # Normalize by 10 minutes

        # Action cost (holding time penalty)
        action_cost = w3 * holding_time / self.max_holding_time

        # Calculate final reward (negative penalties)
        reward = -headway_penalty - wait_time_penalty - action_cost

        # Bonus for good headway regularity
        if abs(bus_headway_fwd - self.target_headway) < 0.1 * self.target_headway:
            reward += 0.5  # Bonus for being close to target

        return reward

    def _calculate_current_headways(self) -> Dict[str, float]:
        """Calculate current headways between buses"""

        headways = {}

        # Sort buses by position on route
        sorted_buses = sorted(self.buses, 
                            key=lambda b: b.current_stop_idx + b.last_departure_time / 3600)

        for i, bus in enumerate(sorted_buses):
            # Forward headway (to next bus)
            next_bus = sorted_buses[(i + 1) % len(sorted_buses)]
            forward_headway = self._calculate_bus_separation(bus, next_bus)
            headways[f"{bus.bus_id}_forward"] = forward_headway

            # Backward headway (from previous bus)
            prev_bus = sorted_buses[i - 1]  # Python handles negative indexing
            backward_headway = self._calculate_bus_separation(prev_bus, bus)
            headways[f"{bus.bus_id}_backward"] = backward_headway

        return headways

    def _calculate_bus_separation(self, bus1: BusAgent, bus2: BusAgent) -> float:
        """Calculate time separation between two buses"""

        # Simplified calculation based on stop indices and departure times
        stop_diff = (bus2.current_stop_idx - bus1.current_stop_idx) % len(self.stops)

        if stop_diff == 0:
            # Same stop - use departure time difference
            return abs(bus2.last_departure_time - bus1.last_departure_time)
        else:
            # Different stops - estimate based on route progress
            estimated_travel_time = stop_diff * 120  # 2 minutes per stop average
            time_diff = bus2.last_departure_time - bus1.last_departure_time
            return max(60, estimated_travel_time + time_diff)  # Minimum 1 minute

    def _calculate_headway_variance(self) -> float:
        """Calculate variance in current headways"""

        headways = self._calculate_current_headways()
        forward_headways = [v for k, v in headways.items() if '_forward' in k]

        if len(forward_headways) < 2:
            return 0

        return np.var(forward_headways)

    def _estimate_passenger_wait_time(self, bus: BusAgent) -> float:
        """Estimate passenger wait time impact"""

        # Simplified estimation - in practice would use queuing theory
        headways = self._calculate_current_headways()
        bus_headway = headways.get(f"{bus.bus_id}_forward", self.target_headway)

        # Average wait time = headway² / (2 * headway) for random arrivals
        estimated_wait = (bus_headway ** 2) / (2 * bus_headway) if bus_headway > 0 else 300

        return min(estimated_wait, 1800)  # Cap at 30 minutes

    def _get_segment_travel_time(self, stop_idx: int) -> float:
        """Get travel time for segment starting at stop_idx"""

        # Base travel time with some randomness for traffic
        base_time = 120  # 2 minutes base
        traffic_factor = np.random.uniform(0.8, 1.5)  # Traffic variability

        return base_time * traffic_factor

    def _get_stop_position(self, stop_idx: int) -> Tuple[float, float]:
        """Get lat/lon position of a stop"""

        if stop_idx < len(self.stops):
            stop = self.stops[stop_idx]
            return (stop.get('lat', 28.6), stop.get('lon', 77.2))

        # Default position if stop not found
        return (28.6, 77.2)

    def _update_passenger_wait_times(self):
        """Update passenger wait time tracking"""

        # Simplified passenger wait time tracking
        for bus in self.buses:
            stop_headway = self._calculate_current_headways().get(f"{bus.bus_id}_forward", 600)
            estimated_passengers_waiting = max(1, stop_headway / 60)  # 1 passenger per minute

            # Add to total wait time
            self.total_passenger_wait_time += estimated_passengers_waiting * (stop_headway / 2)

    def _get_observation(self) -> np.ndarray:
        """Get current observation for the agent"""

        # Get current decision bus
        current_bus = self._get_current_decision_bus()

        if current_bus is None:
            # Return default observation
            return np.array([0.5, 0.5, 0.5, 0.5, 300, 25, 25], dtype=np.float32)

        # Calculate headways
        headways = self._calculate_current_headways()

        # Build observation vector
        obs = np.array([
            current_bus.current_stop_idx / len(self.stops),  # Normalized stop position
            (self.current_time % 86400) / 86400,            # Normalized time of day
            min(1.0, headways.get(f"{current_bus.bus_id}_forward", 600) / 1200),  # Normalized forward headway
            min(1.0, headways.get(f"{current_bus.bus_id}_backward", 600) / 1200), # Normalized backward headway
            self._get_segment_travel_time(current_bus.current_stop_idx),          # Next segment time
            current_bus.passenger_load,                                          # Passenger load
            current_bus.speed                                                    # Current speed
        ], dtype=np.float32)

        return obs

    def _is_episode_done(self) -> bool:
        """Check if episode should end"""

        max_steps = self.simulation_params.get('max_episode_steps', 1000)
        max_time = self.simulation_params.get('max_episode_time', 7200)  # 2 hours

        return (self.episode_step >= max_steps or 
                self.current_time >= max_time)

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for the completed episode"""

        if not self.episode_rewards:
            return {}

        stats = {
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'headway_variance': self._calculate_headway_variance(),
            'total_passenger_wait_time': self.total_passenger_wait_time,
            'total_control_delay': self.total_control_delay,
            'episode_length': self.episode_step
        }

        return stats

    def render(self, mode='human'):
        """Render the environment (optional)"""

        if mode == 'human':
            print(f"Time: {self.current_time:.0f}s, Step: {self.episode_step}")
            print(f"Buses: {len(self.buses)}")

            headways = self._calculate_current_headways()
            forward_headways = [v for k, v in headways.items() if '_forward' in k]

            if forward_headways:
                print(f"Headway variance: {np.var(forward_headways):.1f}")
                print(f"Mean headway: {np.mean(forward_headways):.1f}s")

# Example usage
if __name__ == "__main__":

    # Define route configuration
    route_config = {
        'route_id': 'route_34',
        'num_buses': 5,
        'stops': [
            {'id': 'stop_1', 'lat': 28.60, 'lon': 77.20},
            {'id': 'stop_2', 'lat': 28.61, 'lon': 77.21},
            {'id': 'stop_3', 'lat': 28.62, 'lon': 77.22},
            {'id': 'stop_4', 'lat': 28.63, 'lon': 77.23},
            {'id': 'stop_5', 'lat': 28.64, 'lon': 77.24}
        ]
    }

    simulation_params = {
        'target_headway': 600,
        'max_holding_time': 120,
        'time_step': 30,
        'max_episode_steps': 100
    }

    # Create environment
    env = BusControlEnvironment(route_config, simulation_params)

    print("RL Environment initialized")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test episode
    obs = env.reset()
    total_reward = 0

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"Step {step}: Action={env.action_meanings[action]}, Reward={reward:.2f}")

        if done:
            break

    print(f"Episode finished. Total reward: {total_reward:.2f}")
    print(f"Episode stats: {env.get_episode_stats()}")