# Dynamic Bus Service Rationalization: A Phased Machine Learning Approach

## Overview

This project implements a comprehensive framework for dynamic bus service rationalization designed to combat bus bunching and improve headway regularity in urban transit systems. The implementation is based on the research document "A Framework for Dynamic Bus Service Rationalization: A Phased Machine Learning Approach to Anti-Bunching and Headway Regularity."

## Key Features

- **Phased Implementation**: Progresses through 4 phases from data foundation to advanced RL control
- **Real-time Data Processing**: Integrates GTFS static data with live GTFS-RT feeds
- **Multiple Control Strategies**: 
  - Rule-based baseline controller
  - XGBoost-powered predictive control  
  - Reinforcement Learning optimal control
- **Traffic Integration**: Google Maps API for real-time traffic-aware predictions
- **Event-driven Simulation**: Digital twin environment for RL training and system analysis
- **Comprehensive Evaluation**: Statistical analysis and performance metrics
- **Real-time Dashboard**: Web-based monitoring and visualization

## Architecture

### Core Components

1. **Data Layer** (`data/`)
   - `gtfs_processor.py`: Static GTFS data processing
   - `realtime_archiver.py`: Live data collection and archiving
   - `traffic_api.py`: Google Maps API integration

2. **Models** (`models/`)
   - `baseline_controller.py`: Rule-based bus holding control
   - `xgboost_predictor.py`: ML-based arrival time prediction
   - `rl_environment.py`: OpenAI Gym environment for RL training

3. **Simulation** (`simulation/`)
   - `event_simulator.py`: Event-driven transit system simulation

4. **Evaluation** (`evaluation/`)
   - `metrics.py`: Performance evaluation and statistical analysis

5. **Dashboard** (`dashboard/`)
   - `visualization.py`: Real-time web dashboard

6. **Configuration** (`config/`)
   - `settings.py`: System configuration and parameters

## Installation

### Prerequisites

- Python 3.8+
- SQLite (for data storage)
- Optional: PostgreSQL (for production deployment)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd bus-rationalization
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Initialize database:
```bash
python -c "from config.settings import Config; print('Database initialized')"
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Delhi OTD API Configuration
OTD_API_KEY=your_otd_api_key_here

# Google Maps API Configuration  
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///bus_data.db

# Optional: PostgreSQL for production
# DATABASE_URL=postgresql://user:password@localhost:5432/bus_system
```

### API Keys Setup

1. **Delhi OTD API Key**: 
   - Register at [otd.delhi.gov.in](https://otd.delhi.gov.in)
   - Request API access for real-time data

2. **Google Maps API Key**:
   - Create project in [Google Cloud Console](https://console.cloud.google.com)
   - Enable Maps JavaScript API and Directions API
   - Create API key with appropriate restrictions

## Usage

### Quick Start Demo

Run the complete 4-phase pipeline:

```bash
python main.py --demo
```

This will:
1. Initialize data foundation and baseline controller
2. Set up ML prediction capabilities  
3. Create RL environment and run simulation
4. Generate comparative performance analysis

### Phase-by-Phase Execution

Run individual phases:

```bash
# Phase 1: Data Foundation
python main.py --phase 1

# Phase 2: ML Prediction
python main.py --phase 2

# Phase 3: RL Development  
python main.py --phase 3

# Phase 4: Evaluation
python main.py --phase 4
```

### Real-time Dashboard

Launch the monitoring dashboard:

```bash
python dashboard/visualization.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

### Custom Configuration

Modify `config/settings.py` for custom parameters:

```python
# Control Strategy Configuration
CONTROL_PARAMS = {
    'target_headway': 600,  # 10 minutes
    'headway_tolerance': 0.2,  # 20% tolerance
    'max_holding_time': 120,  # 2 minutes max hold
}

# Model Configuration  
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
}
```

## Key Methodologies

### 1. Rule-Based Baseline Control

Implements the core logic: "If gap ahead < target AND gap behind > target → hold bus"

```python
from models.baseline_controller import RuleBasedController

controller = RuleBasedController(target_headway=600)
decision = controller.make_control_decision(bus_state)
```

### 2. XGBoost Predictive Control

Trains on historical data with features including:
- Temporal patterns (time of day, day of week)
- Headway measurements (forward/backward)
- Traffic conditions (Google Maps API)
- Route characteristics

```python
from models.xgboost_predictor import XGBoostArrivalPredictor

predictor = XGBoostArrivalPredictor()
X, y = predictor.prepare_training_data(days_back=7)
metrics = predictor.train_model(X, y)
```

### 3. Reinforcement Learning Control

Multi-agent RL using PPO algorithm with:
- **State**: Stop position, time, headways, traffic, passenger load
- **Actions**: Hold durations (0s, 30s, 60s, 90s)  
- **Reward**: Balances headway regularity, passenger wait time, control cost

```python
from models.rl_environment import BusControlEnvironment

env = BusControlEnvironment(route_config, sim_params)
# Train with stable-baselines3 PPO
```

### 4. Event-Driven Simulation

Digital twin for RL training and system analysis:

```python
from simulation.event_simulator import EventDrivenTransitSimulator

simulator = EventDrivenTransitSimulator(route_config, sim_params)
results = simulator.run_simulation(duration=3600)
```

## Performance Evaluation

### Key Performance Indicators (KPIs)

1. **Headway Standard Deviation**: Service regularity measure
2. **Mean Passenger Wait Time**: Calculated as E[H²]/(2×E[H])  
3. **Bunching Frequency**: Events where headway < 50% of target
4. **Total Control Delay**: Sum of all holding interventions

### Statistical Analysis

```python
from evaluation.metrics import BusSystemMetrics

metrics = BusSystemMetrics()
comparison = metrics.compare_strategies(baseline_metrics, treatment_metrics)
significance = metrics.statistical_significance_test(baseline_data, treatment_data)
```

### Visualization

The dashboard provides:
- Real-time headway monitoring
- Service quality indicators
- Route map with live bus positions
- Strategy performance comparison
- Statistical confidence intervals

## Data Sources

### Static GTFS Data
- **Source**: Delhi Open Transit Data (OTD) portal
- **Files**: stops.txt, routes.txt, trips.txt, stop_times.txt
- **Purpose**: Network topology and scheduled service

### Real-time GTFS-RT Data  
- **Source**: OTD VehiclePositions.pb endpoint
- **Update**: Every 15-60 seconds
- **Purpose**: Live bus positions and actual headways

### Traffic Data
- **Source**: Google Maps Routes API
- **Purpose**: Traffic-aware travel time predictions
- **Fallback**: Mock API for development/testing

## Project Structure

```
bus_rationalization/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration parameters
├── data/
│   ├── __init__.py
│   ├── gtfs_processor.py     # GTFS data processing
│   ├── realtime_archiver.py  # Live data collection
│   └── traffic_api.py        # Google Maps integration
├── models/
│   ├── __init__.py
│   ├── baseline_controller.py # Rule-based control
│   ├── xgboost_predictor.py  # ML arrival prediction
│   └── rl_environment.py     # RL training environment
├── simulation/
│   ├── __init__.py
│   ├── event_simulator.py    # Event-driven simulation
│   └── digital_twin.py       # System digital twin
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # Performance evaluation
├── dashboard/
│   ├── __init__.py
│   └── visualization.py      # Web dashboard
└── utils/
    ├── __init__.py
    └── helpers.py            # Utility functions
```

## Development Guidelines

### Code Organization

- **Modular Design**: Each component is independently testable
- **Configuration-Driven**: Parameters externalized to config files
- **Logging**: Comprehensive logging throughout the system
- **Error Handling**: Graceful degradation and recovery

### Testing

Run unit tests:

```bash
python -m pytest tests/
```

Generate test data:

```bash
python utils/helpers.py  # Test utility functions
python -c "from utils.helpers import generate_sample_gtfs_data; print('Test data generated')"
```

### Extension Points

1. **Additional Control Strategies**: Extend `baseline_controller.py`
2. **New ML Models**: Add to `models/` directory
3. **Custom Metrics**: Extend `evaluation/metrics.py`  
4. **Dashboard Components**: Add to `dashboard/visualization.py`

## Deployment

### Production Considerations

1. **Database**: Migrate from SQLite to PostgreSQL for production
2. **API Rate Limits**: Implement proper rate limiting for external APIs
3. **Monitoring**: Set up logging aggregation and alerting
4. **Scalability**: Consider containerization with Docker

### Docker Deployment (Optional)

```bash
# Create Dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py", "--demo"]
```

## Research Background

This implementation is based on the research framework that identifies bus bunching as a systemic failure caused by positive feedback loops in transit operations. The solution employs:

1. **Systems Approach**: Treats bunching as emergent behavior requiring system-level intervention
2. **Data-Driven Control**: Uses real-time data to make optimal control decisions  
3. **Phased Implementation**: Delivers value incrementally while building capability
4. **Rigorous Evaluation**: Employs statistical methods to validate improvements

### Key Insights

- **Headway Regularity > Schedule Adherence**: Focus on relative spacing rather than absolute timing
- **Predictive Control > Reactive**: Use ML to anticipate problems before they occur
- **Holistic Optimization**: RL discovers non-obvious control strategies
- **Digital Twin Value**: Simulation enables risk-free experimentation

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```
   Error: Invalid API key
   Solution: Check .env file and API key validity
   ```

2. **Database Connection**:
   ```
   Error: Database locked
   Solution: Close other connections or restart application
   ```

3. **Missing Dependencies**:
   ```
   Error: ModuleNotFoundError
   Solution: pip install -r requirements.txt
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document classes and methods with docstrings
- Maintain test coverage above 80%

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bus_rationalization_2024,
  title={Dynamic Bus Service Rationalization: A Phased Machine Learning Approach},
  author={Gourav Sahu},
  year={2025},
  howpublished={\url{https://github.com/binaryecheos/dtc-route-rationalization}}
}
```

## Acknowledgments

- Delhi Government for providing open transit data through OTD portal
- Google Maps API for traffic data integration
- OpenAI Gym for RL environment framework
- The transit research community for foundational work on bus bunching

## Contact

For questions or support, please contact:
- Email: gourav.sahu.1695@gmail.com
- GitHub: @binaryecheos

## Roadmap

### Upcoming Features

- [ ] Multi-route optimization
- [ ] Passenger demand forecasting
- [ ] Weather impact modeling
- [ ] Mobile app integration
- [ ] Real-time passenger information system

### Research Extensions

- [ ] Deep RL with attention mechanisms
- [ ] Federated learning across cities
- [ ] Causal inference for intervention analysis
- [ ] Integration with autonomous vehicle systems

---

