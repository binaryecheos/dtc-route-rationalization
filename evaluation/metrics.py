import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import logging

class BusSystemMetrics:
    """
    Comprehensive evaluation metrics for bus system performance.
    Implements the KPIs outlined in the research document.
    """

    def __init__(self, database_url: str = "sqlite:///bus_data.db"):
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)

    def calculate_headway_statistics(self, route_id: str, 
                                   time_window: Tuple[datetime, datetime]) -> Dict[str, float]:
        """
        Calculate headway regularity metrics from historical data
        """
        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))

            # Convert time window to timestamps
            start_ts = int(time_window[0].timestamp())
            end_ts = int(time_window[1].timestamp())

            # Get vehicle position data
            query = '''
                SELECT timestamp, vehicle_id, latitude, longitude
                FROM vehicle_positions 
                WHERE route_id = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''

            df = pd.read_sql_query(query, conn, params=[route_id, start_ts, end_ts])
            conn.close()

            if df.empty:
                return {}

            # Calculate headways (simplified approach)
            # In practice, would need more sophisticated stop detection
            headways = self._calculate_headways_from_positions(df)

            if not headways:
                return {}

            # Calculate statistics
            stats = {
                'mean_headway': np.mean(headways),
                'std_headway': np.std(headways),
                'min_headway': np.min(headways),
                'max_headway': np.max(headways),
                'cv_headway': np.std(headways) / np.mean(headways) if np.mean(headways) > 0 else 0,
                'headway_count': len(headways)
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to calculate headway statistics: {e}")
            return {}

    def calculate_passenger_wait_time(self, headways: List[float]) -> Dict[str, float]:
        """
        Calculate estimated passenger wait times from headway data
        Uses the formula: E[W] = E[H²] / (2 * E[H]) for random arrivals
        """

        if not headways:
            return {}

        mean_headway = np.mean(headways)
        mean_headway_squared = np.mean([h**2 for h in headways])

        # Calculate expected wait time for random arrivals
        if mean_headway > 0:
            expected_wait = mean_headway_squared / (2 * mean_headway)
        else:
            expected_wait = 0

        # Additional statistics
        metrics = {
            'expected_wait_time': expected_wait,
            'max_wait_time': np.max(headways),  # Worst case
            'wait_time_variance': self._calculate_wait_variance(headways),
            'service_regularity_index': mean_headway / np.std(headways) if np.std(headways) > 0 else 0
        }

        return metrics

    def calculate_bunching_frequency(self, headways: List[float], 
                                   target_headway: float = 600,
                                   bunching_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate bunching event frequency and severity
        """

        if not headways:
            return {}

        # Define bunching as headway < threshold * target
        bunching_limit = target_headway * bunching_threshold

        # Count bunching events
        bunching_events = [h for h in headways if h < bunching_limit]

        # Calculate severity (how far below threshold)
        if bunching_events:
            avg_bunching_severity = np.mean([(bunching_limit - h) / bunching_limit 
                                           for h in bunching_events])
        else:
            avg_bunching_severity = 0

        metrics = {
            'bunching_frequency': len(bunching_events) / len(headways) if headways else 0,
            'bunching_events_count': len(bunching_events),
            'total_observations': len(headways),
            'avg_bunching_severity': avg_bunching_severity,
            'most_severe_bunching': (bunching_limit - min(bunching_events)) / bunching_limit if bunching_events else 0
        }

        return metrics

    def calculate_control_delay_metrics(self, control_actions: List[Dict]) -> Dict[str, float]:
        """
        Calculate metrics related to control intervention costs
        """

        if not control_actions:
            return {}

        # Extract holding times
        holding_times = [action.get('holding_time', 0) for action in control_actions]

        # Calculate statistics
        total_holding = sum(holding_times)
        num_holds = sum(1 for h in holding_times if h > 0)

        metrics = {
            'total_control_delay': total_holding,
            'avg_holding_time': total_holding / max(num_holds, 1),
            'holding_frequency': num_holds / len(control_actions) if control_actions else 0,
            'max_holding_time': max(holding_times) if holding_times else 0,
            'control_efficiency': self._calculate_control_efficiency(control_actions)
        }

        return metrics

    def compare_strategies(self, baseline_metrics: Dict[str, float],
                         treatment_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compare performance between baseline and treatment strategies
        """

        comparison = {}

        # Compare key metrics
        key_metrics = [
            'mean_headway', 'std_headway', 'cv_headway',
            'expected_wait_time', 'bunching_frequency'
        ]

        for metric in key_metrics:
            if metric in baseline_metrics and metric in treatment_metrics:
                baseline_val = baseline_metrics[metric]
                treatment_val = treatment_metrics[metric]

                if baseline_val != 0:
                    # Calculate percentage change
                    pct_change = (treatment_val - baseline_val) / abs(baseline_val) * 100
                    comparison[f'{metric}_pct_change'] = pct_change

                    # Calculate improvement (negative change is good for these metrics)
                    if metric in ['std_headway', 'cv_headway', 'expected_wait_time', 'bunching_frequency']:
                        improvement = -pct_change  # Lower is better
                    else:
                        improvement = pct_change   # Higher is better (e.g., headway adherence)

                    comparison[f'{metric}_improvement'] = improvement

        # Overall performance score
        improvement_scores = [v for k, v in comparison.items() if '_improvement' in k]
        if improvement_scores:
            comparison['overall_improvement'] = np.mean(improvement_scores)

        return comparison

    def statistical_significance_test(self, baseline_data: List[float],
                                    treatment_data: List[float],
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test statistical significance of performance differences
        """

        if len(baseline_data) < 2 or len(treatment_data) < 2:
            return {'error': 'Insufficient data for statistical test'}

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline_data, treatment_data)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(baseline_data) - 1) * np.var(baseline_data, ddof=1) +
                             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                            (len(baseline_data) + len(treatment_data) - 2))

        cohens_d = (np.mean(treatment_data) - np.mean(baseline_data)) / pooled_std

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(baseline_data, treatment_data, 
                                              alternative='two-sided')

        results = {
            't_statistic': t_stat,
            't_test_p_value': p_value,
            'is_significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_p_value,
            'baseline_mean': np.mean(baseline_data),
            'treatment_mean': np.mean(treatment_data),
            'baseline_std': np.std(baseline_data),
            'treatment_std': np.std(treatment_data)
        }

        return results

    def generate_performance_report(self, route_id: str, 
                                  evaluation_period: Tuple[datetime, datetime],
                                  control_strategy: str = "unknown") -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """

        report = {
            'route_id': route_id,
            'evaluation_period': {
                'start': evaluation_period[0].isoformat(),
                'end': evaluation_period[1].isoformat()
            },
            'control_strategy': control_strategy,
            'generated_at': datetime.now().isoformat()
        }

        # Calculate headway metrics
        headway_stats = self.calculate_headway_statistics(route_id, evaluation_period)
        report['headway_metrics'] = headway_stats

        if headway_stats and 'mean_headway' in headway_stats:
            # Simulate headway data for demonstration
            mean_hw = headway_stats['mean_headway']
            std_hw = headway_stats['std_headway']
            n_samples = headway_stats['headway_count']

            # Generate sample headway data (would use real data in practice)
            sample_headways = np.random.normal(mean_hw, std_hw, max(n_samples, 10))
            sample_headways = [max(60, h) for h in sample_headways]  # Minimum 1 minute

            # Calculate derived metrics
            report['passenger_wait_metrics'] = self.calculate_passenger_wait_time(sample_headways)
            report['bunching_metrics'] = self.calculate_bunching_frequency(sample_headways)

            # Service quality assessment
            report['service_quality'] = self._assess_service_quality(headway_stats)

        return report

    def _calculate_headways_from_positions(self, df: pd.DataFrame) -> List[float]:
        """
        Calculate headways from vehicle position data
        Simplified implementation - would need more sophisticated logic in practice
        """

        headways = []

        # Group by vehicle and calculate time gaps
        for vehicle_id in df['vehicle_id'].unique():
            vehicle_data = df[df['vehicle_id'] == vehicle_id].sort_values('timestamp')

            if len(vehicle_data) > 1:
                time_diffs = vehicle_data['timestamp'].diff().dropna()
                # Filter out very short intervals (position updates)
                meaningful_gaps = [gap for gap in time_diffs if gap > 300]  # > 5 minutes
                headways.extend(meaningful_gaps)

        return headways

    def _calculate_wait_variance(self, headways: List[float]) -> float:
        """Calculate variance in passenger wait times"""

        # Simplified calculation
        # For random arrivals, wait time variance depends on headway variance
        headway_var = np.var(headways)
        mean_headway = np.mean(headways)

        if mean_headway > 0:
            # Approximation based on queuing theory
            wait_variance = (headway_var + mean_headway**2) / 12
        else:
            wait_variance = 0

        return wait_variance

    def _calculate_control_efficiency(self, control_actions: List[Dict]) -> float:
        """Calculate efficiency of control interventions"""

        if not control_actions:
            return 0

        # Simplified efficiency metric
        # In practice, would correlate control actions with resulting improvements

        total_actions = len(control_actions)
        effective_actions = sum(1 for action in control_actions 
                              if action.get('resulted_in_improvement', False))

        return effective_actions / total_actions if total_actions > 0 else 0

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""

        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def _assess_service_quality(self, headway_stats: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall service quality based on headway performance"""

        cv = headway_stats.get('cv_headway', 0)

        # Service quality thresholds (based on transit industry standards)
        if cv < 0.2:
            quality_level = "excellent"
            quality_score = 5
        elif cv < 0.3:
            quality_level = "good"  
            quality_score = 4
        elif cv < 0.4:
            quality_level = "acceptable"
            quality_score = 3
        elif cv < 0.6:
            quality_level = "poor"
            quality_score = 2
        else:
            quality_level = "very poor"
            quality_score = 1

        return {
            'quality_level': quality_level,
            'quality_score': quality_score,
            'coefficient_of_variation': cv,
            'assessment_criteria': 'Based on headway coefficient of variation'
        }

    def plot_headway_distribution(self, headways: List[float], 
                                 target_headway: float = 600,
                                 title: str = "Headway Distribution") -> plt.Figure:
        """Create visualization of headway distribution"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(headways, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(target_headway, color='red', linestyle='--', 
                   label=f'Target: {target_headway}s')
        ax1.axvline(np.mean(headways), color='green', linestyle='-', 
                   label=f'Mean: {np.mean(headways):.1f}s')
        ax1.set_xlabel('Headway (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Histogram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(headways, vert=True)
        ax2.axhline(target_headway, color='red', linestyle='--', 
                   label=f'Target: {target_headway}s')
        ax2.set_ylabel('Headway (seconds)')
        ax2.set_title(f'{title} - Box Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_performance_dashboard(self, metrics_data: Dict[str, Any]) -> plt.Figure:
        """Create performance dashboard visualization"""

        fig = plt.figure(figsize=(15, 10))

        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Headway statistics
        ax1 = fig.add_subplot(gs[0, 0])
        headway_metrics = metrics_data.get('headway_metrics', {})
        if headway_metrics:
            metrics_names = ['Mean', 'Std', 'CV']
            values = [
                headway_metrics.get('mean_headway', 0) / 60,  # Convert to minutes
                headway_metrics.get('std_headway', 0) / 60,
                headway_metrics.get('cv_headway', 0)
            ]
            ax1.bar(metrics_names, values, color=['blue', 'orange', 'green'])
            ax1.set_title('Headway Statistics')
            ax1.set_ylabel('Minutes / Ratio')

        # Service quality indicator
        ax2 = fig.add_subplot(gs[0, 1])
        service_quality = metrics_data.get('service_quality', {})
        if service_quality:
            quality_score = service_quality.get('quality_score', 0)
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            color = colors[min(quality_score - 1, 4)] if quality_score > 0 else 'gray'

            ax2.bar(['Service Quality'], [quality_score], color=color)
            ax2.set_ylim(0, 5)
            ax2.set_title('Service Quality Score')
            ax2.set_ylabel('Score (1-5)')

        # Bunching frequency
        ax3 = fig.add_subplot(gs[0, 2])
        bunching_metrics = metrics_data.get('bunching_metrics', {})
        if bunching_metrics:
            bunching_freq = bunching_metrics.get('bunching_frequency', 0) * 100
            ax3.bar(['Bunching'], [bunching_freq], color='red', alpha=0.7)
            ax3.set_title('Bunching Frequency')
            ax3.set_ylabel('Percentage (%)')

        # Add summary text
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        summary_text = self._generate_summary_text(metrics_data)
        ax4.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center',
                transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.5))

        plt.suptitle(f"Bus System Performance Dashboard", fontsize=16, fontweight='bold')

        return fig

    def _generate_summary_text(self, metrics_data: Dict[str, Any]) -> str:
        """Generate summary text for dashboard"""

        lines = []

        # Service quality
        service_quality = metrics_data.get('service_quality', {})
        if service_quality:
            quality_level = service_quality.get('quality_level', 'unknown')
            lines.append(f"Service Quality: {quality_level.title()}")

        # Key metrics
        headway_metrics = metrics_data.get('headway_metrics', {})
        if headway_metrics:
            mean_hw = headway_metrics.get('mean_headway', 0)
            cv = headway_metrics.get('cv_headway', 0)
            lines.append(f"Average Headway: {mean_hw/60:.1f} minutes")
            lines.append(f"Headway Variability (CV): {cv:.2f}")

        bunching_metrics = metrics_data.get('bunching_metrics', {})
        if bunching_metrics:
            bunching_freq = bunching_metrics.get('bunching_frequency', 0)
            lines.append(f"Bunching Rate: {bunching_freq*100:.1f}%")

        passenger_metrics = metrics_data.get('passenger_wait_metrics', {})
        if passenger_metrics:
            wait_time = passenger_metrics.get('expected_wait_time', 0)
            lines.append(f"Expected Wait Time: {wait_time/60:.1f} minutes")

        return "\n".join(lines) if lines else "No metrics available"

# Example usage
if __name__ == "__main__":

    metrics = BusSystemMetrics()

    # Generate sample data for demonstration
    np.random.seed(42)

    # Sample headway data (in seconds)
    baseline_headways = np.random.normal(600, 120, 100)  # Target 10 min ± 2 min
    baseline_headways = [max(60, h) for h in baseline_headways]  # Minimum 1 minute

    treatment_headways = np.random.normal(600, 80, 100)   # Improved variability
    treatment_headways = [max(60, h) for h in treatment_headways]

    print("Bus System Metrics Evaluation")
    print("="*50)

    # Calculate baseline metrics
    baseline_stats = {
        'mean_headway': np.mean(baseline_headways),
        'std_headway': np.std(baseline_headways),
        'cv_headway': np.std(baseline_headways) / np.mean(baseline_headways)
    }

    print("\nBaseline Performance:")
    for metric, value in baseline_stats.items():
        print(f"{metric}: {value:.2f}")

    # Calculate treatment metrics  
    treatment_stats = {
        'mean_headway': np.mean(treatment_headways),
        'std_headway': np.std(treatment_headways),
        'cv_headway': np.std(treatment_headways) / np.mean(treatment_headways)
    }

    print("\nTreatment Performance:")
    for metric, value in treatment_stats.items():
        print(f"{metric}: {value:.2f}")

    # Compare strategies
    comparison = metrics.compare_strategies(baseline_stats, treatment_stats)

    print("\nPerformance Comparison:")
    for metric, value in comparison.items():
        if 'improvement' in metric:
            print(f"{metric}: {value:.2f}%")

    # Statistical significance
    sig_test = metrics.statistical_significance_test(baseline_headways, treatment_headways)

    print("\nStatistical Analysis:")
    print(f"P-value: {sig_test['t_test_p_value']:.4f}")
    print(f"Significant: {sig_test['is_significant']}")
    print(f"Effect size: {sig_test['effect_size_interpretation']}")