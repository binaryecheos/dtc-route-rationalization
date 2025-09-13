import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Delhi OTD API Configuration
    OTD_BASE_URL = "https://otd.delhi.gov.in"
    REALTIME_API_KEY = "L5jVvGl6iLEdSqnZG42pEm5LD94t1PYF"
    OTD_API_KEY = REALTIME_API_KEY
    GTFS_RT_ENDPOINT = "/api/realtime/VehiclePositions.pb"

    # Google Maps API Configuration
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "your_google_api_key_here")

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bus_data.db")

    # Data Archiving Configuration
    ARCHIVE_FREQUENCY = 30  # seconds between data collection
    MAX_ARCHIVE_SIZE = 10000  # maximum records before cleanup

    # Model Configuration
    XGBOOST_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # RL Configuration
    RL_PARAMS = {
        'algorithm': 'PPO',
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'clip_range': 0.2
    }

    # Simulation Configuration
    SIM_PARAMS = {
        'time_step': 1.0,  # simulation time step in seconds
        'warm_up_time': 3600,  # warm-up period in seconds
        'total_sim_time': 86400,  # total simulation time (24 hours)
        'random_seed': 42
    }

    # Control Strategy Configuration
    CONTROL_PARAMS = {
        'target_headway': 600,  # 10 minutes in seconds
        'headway_tolerance': 0.2,  # 20% tolerance
        'max_holding_time': 120,  # maximum holding time in seconds
        'min_headway_ratio': 0.5,  # minimum headway as ratio of target
        'max_headway_ratio': 2.0   # maximum headway as ratio of target
    }

    # Dashboard Configuration
    DASHBOARD_CONFIG = {
        'update_interval': 30,  # seconds
        'map_center': [28.6139, 77.2090],  # Delhi coordinates
        'map_zoom': 11
    }

    # Evaluation Metrics
    METRICS_CONFIG = {
        'evaluation_window': 3600,  # 1 hour evaluation windows
        'statistical_confidence': 0.95,
        'performance_threshold': 0.15  # 15% improvement threshold
    }