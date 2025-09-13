import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class ControlDecision:
    """Represents a control decision for a bus"""
    bus_id: str
    stop_id: str
    action: str  # 'depart', 'hold_30s', 'hold_60s', 'hold_90s'
    reason: str
    confidence: float
    timestamp: datetime

@dataclass
class BusState:
    """Current state of a bus"""
    bus_id: str
    route_id: str
    current_stop: str
    latitude: float
    longitude: float
    passenger_load: Optional[int]
    scheduled_departure: datetime
    actual_arrival: datetime
    headway_forward: Optional[float]  # seconds to bus ahead
    headway_backward: Optional[float]  # seconds from bus behind

class RuleBasedController:
    """
    Implements rule-based bus holding control to prevent bunching.
    This serves as the baseline against which ML models are compared.
    """

    def __init__(self, target_headway: int = 600, headway_tolerance: float = 0.2,
                 max_holding_time: int = 120):
        """
        Initialize controller with operational parameters

        Args:
            target_headway: Target headway in seconds (default 10 minutes)
            headway_tolerance: Acceptable deviation as fraction of target
            max_holding_time: Maximum time to hold a bus in seconds
        """
        self.target_headway = target_headway
        self.headway_tolerance = headway_tolerance
        self.max_holding_time = max_holding_time
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.decisions_made = 0
        self.holds_applied = 0
        self.total_hold_time = 0

    def analyze_headway_situation(self, bus_state: BusState) -> Dict[str, float]:
        """Analyze current headway situation for a bus"""

        analysis = {
            'forward_headway_ratio': 0.0,
            'backward_headway_ratio': 0.0,
            'is_bunching': False,
            'has_large_gap_behind': False,
            'bunching_severity': 0.0
        }

        if bus_state.headway_forward is not None:
            analysis['forward_headway_ratio'] = bus_state.headway_forward / self.target_headway

        if bus_state.headway_backward is not None:
            analysis['backward_headway_ratio'] = bus_state.headway_backward / self.target_headway

        # Detect bunching: forward headway too small
        if bus_state.headway_forward is not None:
            min_acceptable = self.target_headway * (1 - self.headway_tolerance)
            if bus_state.headway_forward < min_acceptable:
                analysis['is_bunching'] = True
                analysis['bunching_severity'] = (min_acceptable - bus_state.headway_forward) / min_acceptable

        # Detect large gap behind: backward headway too large
        if bus_state.headway_backward is not None:
            max_acceptable = self.target_headway * (1 + self.headway_tolerance)
            if bus_state.headway_backward > max_acceptable:
                analysis['has_large_gap_behind'] = True

        return analysis

    def make_control_decision(self, bus_state: BusState) -> ControlDecision:
        """
        Make control decision based on rule-based logic

        Core rule: If gap ahead < target AND gap behind > target → hold bus
        """

        analysis = self.analyze_headway_situation(bus_state)
        action = 'depart'
        reason = 'Normal operation'
        confidence = 1.0

        # Rule 1: Basic bunching prevention
        if analysis['is_bunching'] and analysis['has_large_gap_behind']:

            # Determine holding time based on severity
            if analysis['bunching_severity'] > 0.5:
                action = 'hold_90s'
                reason = f"Severe bunching detected (severity: {analysis['bunching_severity']:.2f})"
                confidence = 0.9
            elif analysis['bunching_severity'] > 0.3:
                action = 'hold_60s'  
                reason = f"Moderate bunching detected (severity: {analysis['bunching_severity']:.2f})"
                confidence = 0.8
            else:
                action = 'hold_30s'
                reason = f"Mild bunching detected (severity: {analysis['bunching_severity']:.2f})"
                confidence = 0.7

        # Rule 2: Prevent excessive holding
        elif analysis['forward_headway_ratio'] > 1.5:
            # If we're already far behind, don't hold further
            action = 'depart'
            reason = "Already running behind schedule"
            confidence = 0.9

        # Rule 3: Rush hour adjustments
        current_hour = datetime.now().hour
        if current_hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
            if action.startswith('hold'):
                # Reduce holding time during rush hours
                if action == 'hold_90s':
                    action = 'hold_60s'
                elif action == 'hold_60s':
                    action = 'hold_30s'
                reason += " (adjusted for rush hour)"

        decision = ControlDecision(
            bus_id=bus_state.bus_id,
            stop_id=bus_state.current_stop,
            action=action,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.now()
        )

        # Update performance tracking
        self.decisions_made += 1
        if action != 'depart':
            self.holds_applied += 1
            hold_time = int(action.split('_')[1].replace('s', ''))
            self.total_hold_time += hold_time

        return decision

    def batch_control_decisions(self, bus_states: List[BusState]) -> List[ControlDecision]:
        """Make control decisions for multiple buses"""
        decisions = []

        for bus_state in bus_states:
            decision = self.make_control_decision(bus_state)
            decisions.append(decision)

        return decisions

    def calculate_headways_from_positions(self, vehicle_positions: List[Dict], 
                                        route_id: str) -> Dict[str, float]:
        """
        Calculate current headways from vehicle position data

        This is a simplified implementation - in practice would need more
        sophisticated logic to determine when buses pass specific stops.
        """

        # Filter positions for the specific route
        route_positions = [pos for pos in vehicle_positions if pos.get('route_id') == route_id]

        if len(route_positions) < 2:
            return {}

        # Sort by position along route (simplified - would need route geometry)
        # For now, sort by longitude as a proxy
        route_positions.sort(key=lambda x: x['longitude'])

        headways = {}

        for i in range(len(route_positions)):
            current_bus = route_positions[i]
            bus_id = current_bus['vehicle_id']

            # Calculate forward headway (to bus ahead)
            if i < len(route_positions) - 1:
                next_bus = route_positions[i + 1]
                # Simplified distance calculation - would use route geometry in practice
                distance = abs(next_bus['longitude'] - current_bus['longitude'])
                # Convert to time (rough estimate - 30 km/h average speed)
                forward_headway = distance * 111000 / (30 * 1000 / 3600)  # Convert to seconds
                headways[f"{bus_id}_forward"] = forward_headway

            # Calculate backward headway (from bus behind)  
            if i > 0:
                prev_bus = route_positions[i - 1]
                distance = abs(current_bus['longitude'] - prev_bus['longitude'])
                backward_headway = distance * 111000 / (30 * 1000 / 3600)
                headways[f"{bus_id}_backward"] = backward_headway

        return headways

    def get_performance_stats(self) -> Dict[str, float]:
        """Get controller performance statistics"""

        if self.decisions_made == 0:
            return {}

        stats = {
            'total_decisions': self.decisions_made,
            'holds_applied': self.holds_applied,
            'hold_rate': self.holds_applied / self.decisions_made,
            'total_hold_time_seconds': self.total_hold_time,
            'average_hold_time': self.total_hold_time / max(self.holds_applied, 1)
        }

        return stats

    def reset_performance_stats(self):
        """Reset performance tracking counters"""
        self.decisions_made = 0
        self.holds_applied = 0
        self.total_hold_time = 0

    def simulate_control_impact(self, historical_headways: List[float]) -> Dict[str, float]:
        """
        Simulate the impact of rule-based control on historical headway data
        """

        # Calculate baseline metrics
        baseline_mean = np.mean(historical_headways)
        baseline_std = np.std(historical_headways)
        baseline_cv = baseline_std / baseline_mean if baseline_mean > 0 else 0

        # Simulate control effect (simplified)
        # Rule-based control typically reduces variance by 10-25%
        controlled_headways = []

        for headway in historical_headways:
            # Apply control logic
            if headway < self.target_headway * 0.7:  # Too short
                # Simulate holding effect - increase headway
                adjusted = headway * 1.2
            elif headway > self.target_headway * 1.5:  # Too long
                # Can't fix large gaps with holding alone
                adjusted = headway * 0.95
            else:
                adjusted = headway

            controlled_headways.append(adjusted)

        # Calculate controlled metrics
        controlled_mean = np.mean(controlled_headways)
        controlled_std = np.std(controlled_headways)
        controlled_cv = controlled_std / controlled_mean if controlled_mean > 0 else 0

        # Calculate bunching frequency (headways < 50% of target)
        bunching_threshold = self.target_headway * 0.5
        baseline_bunching = sum(1 for h in historical_headways if h < bunching_threshold)
        controlled_bunching = sum(1 for h in controlled_headways if h < bunching_threshold)

        return {
            'baseline_mean_headway': baseline_mean,
            'controlled_mean_headway': controlled_mean,
            'baseline_std_headway': baseline_std,
            'controlled_std_headway': controlled_std,
            'baseline_cv': baseline_cv,
            'controlled_cv': controlled_cv,
            'variance_reduction': (baseline_std - controlled_std) / baseline_std if baseline_std > 0 else 0,
            'baseline_bunching_events': baseline_bunching,
            'controlled_bunching_events': controlled_bunching,
            'bunching_reduction': (baseline_bunching - controlled_bunching) / max(baseline_bunching, 1)
        }

# Example usage and testing
if __name__ == "__main__":

    # Initialize controller
    controller = RuleBasedController(target_headway=600, headway_tolerance=0.2)

    # Create sample bus state
    bus_state = BusState(
        bus_id="DL01PC1234",
        route_id="34",
        current_stop="stop_123",
        latitude=28.6139,
        longitude=77.2090,
        passenger_load=25,
        scheduled_departure=datetime.now(),
        actual_arrival=datetime.now() - timedelta(minutes=2),
        headway_forward=300,  # 5 minutes (too short for 10 min target)
        headway_backward=900  # 15 minutes (acceptable)
    )

    # Make control decision
    decision = controller.make_control_decision(bus_state)

    print(f"Control Decision:")
    print(f"Bus: {decision.bus_id}")
    print(f"Action: {decision.action}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence}")

    # Test simulation
    sample_headways = [300, 450, 600, 800, 200, 1200, 400, 650, 350, 900]
    simulation_results = controller.simulate_control_impact(sample_headways)

    print(f"\nSimulation Results:")
    print(f"Variance reduction: {simulation_results['variance_reduction']:.2%}")
    print(f"Bunching reduction: {simulation_results['bunching_reduction']:.2%}")