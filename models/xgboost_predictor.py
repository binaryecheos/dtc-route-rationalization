import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from datetime import datetime, timedelta
import sqlite3
import time

class XGBoostArrivalPredictor:
    """
    XGBoost-based model for predicting bus arrival times at upcoming stops.
    Uses historical GTFS-RT data and real-time traffic information.
    """

    def __init__(self, database_url: str = "sqlite:///bus_data.db"):
        self.database_url = database_url
        self.model = None
        self.feature_columns = []
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)

        # Model parameters (based on document recommendations)
        self.model_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp data"""

        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute  
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Peak hour indicators
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    def calculate_headway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate headway and spacing features"""

        # Sort by route, stop, and timestamp for headway calculation
        df_sorted = df.sort_values(['route_id', 'stop_id', 'timestamp'])

        # Calculate time since previous bus at same stop
        df_sorted['prev_bus_time'] = df_sorted.groupby(['route_id', 'stop_id'])['timestamp'].shift(1)
        df_sorted['headway_backward'] = df_sorted['timestamp'] - df_sorted['prev_bus_time']

        # Calculate time to next bus at same stop
        df_sorted['next_bus_time'] = df_sorted.groupby(['route_id', 'stop_id'])['timestamp'].shift(-1)
        df_sorted['headway_forward'] = df_sorted['next_bus_time'] - df_sorted['timestamp']

        # Fill NaN values
        df_sorted['headway_backward'] = df_sorted['headway_backward'].fillna(600)  # Default 10 min
        df_sorted['headway_forward'] = df_sorted['headway_forward'].fillna(600)

        return df_sorted

    def calculate_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate speed and movement features"""

        # Sort by vehicle and timestamp for speed calculation
        df_sorted = df.sort_values(['vehicle_id', 'timestamp'])

        # Calculate distance between consecutive positions
        df_sorted['prev_lat'] = df_sorted.groupby('vehicle_id')['latitude'].shift(1)
        df_sorted['prev_lon'] = df_sorted.groupby('vehicle_id')['longitude'].shift(1)
        df_sorted['prev_time'] = df_sorted.groupby('vehicle_id')['timestamp'].shift(1)

        # Haversine distance calculation (simplified)
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance in meters between two lat/lon points"""
            R = 6371000  # Earth radius in meters

            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            delta_lat = np.radians(lat2 - lat1)
            delta_lon = np.radians(lon2 - lon1)

            a = (np.sin(delta_lat/2)**2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

            return R * c

        # Calculate distance and speed
        mask = df_sorted[['prev_lat', 'prev_lon', 'prev_time']].notna().all(axis=1)

        df_sorted.loc[mask, 'distance_m'] = haversine_distance(
            df_sorted.loc[mask, 'prev_lat'],
            df_sorted.loc[mask, 'prev_lon'], 
            df_sorted.loc[mask, 'latitude'],
            df_sorted.loc[mask, 'longitude']
        )

        df_sorted.loc[mask, 'time_diff'] = (
            df_sorted.loc[mask, 'timestamp'] - df_sorted.loc[mask, 'prev_time']
        )

        # Calculate speed (m/s)
        df_sorted.loc[mask, 'speed_ms'] = (
            df_sorted.loc[mask, 'distance_m'] / 
            df_sorted.loc[mask, 'time_diff'].clip(lower=1)  # Avoid division by zero
        )

        # Convert to km/h and fill NaN
        df_sorted['speed_kmh'] = (df_sorted['speed_ms'] * 3.6).fillna(25)  # Default 25 km/h

        # Recent speed features (rolling averages)
        df_sorted['avg_speed_5min'] = (
            df_sorted.groupby('vehicle_id')['speed_kmh']
            .rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        )

        return df_sorted

    def prepare_training_data(self, days_back: int = 7) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from archived vehicle positions"""

        try:
            conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))

            # Get historical data
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (days_back * 24 * 3600)

            query = '''
                SELECT timestamp, vehicle_id, trip_id, route_id, 
                       latitude, longitude, speed
                FROM vehicle_positions 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''

            df = pd.read_sql_query(query, conn, params=[start_time, end_time])
            conn.close()

            if df.empty:
                raise ValueError("No training data available")

            # Extract features
            df = self.extract_temporal_features(df)
            df = self.calculate_headway_features(df)  
            df = self.calculate_speed_features(df)

            # Create target variable (travel time to next position)
            df_sorted = df.sort_values(['vehicle_id', 'timestamp'])
            df_sorted['next_timestamp'] = df_sorted.groupby('vehicle_id')['timestamp'].shift(-1)
            df_sorted['travel_time_target'] = df_sorted['next_timestamp'] - df_sorted['timestamp']

            # Remove rows without target
            df_sorted = df_sorted.dropna(subset=['travel_time_target'])

            # Feature selection
            feature_cols = [
                'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                'day_sin', 'day_cos', 'is_peak_hour', 'is_weekend',
                'headway_backward', 'headway_forward', 'speed_kmh', 'avg_speed_5min'
            ]

            # Add categorical features with encoding
            categorical_cols = ['route_id']
            for col in categorical_cols:
                if col in df_sorted.columns:
                    le = LabelEncoder()
                    df_sorted[f'{col}_encoded'] = le.fit_transform(df_sorted[col].astype(str))
                    feature_cols.append(f'{col}_encoded')
                    self.label_encoders[col] = le

            # Prepare final dataset
            X = df_sorted[feature_cols].fillna(0)
            y = df_sorted['travel_time_target'].clip(10, 1800)  # Clip to reasonable range

            self.feature_columns = feature_cols

            self.logger.info(f"Prepared training data: {len(X)} samples, {len(feature_cols)} features")

            return X, y

        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            raise

    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, tune_hyperparameters: bool = False) -> Dict[str, float]:
        """Train XGBoost model"""

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )

            if tune_hyperparameters:
                # Hyperparameter tuning
                param_grid = {
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.8, 0.9, 1.0]
                }

                self.model = xgb.XGBRegressor(**self.model_params)

                grid_search = GridSearchCV(
                    self.model, param_grid, cv=3, scoring='neg_mean_absolute_error',
                    n_jobs=-1, verbose=1
                )

                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_

                self.logger.info(f"Best parameters: {grid_search.best_params_}")

            else:
                # Train with default parameters
                self.model = xgb.XGBRegressor(**self.model_params)
                self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)

            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }

            self.logger.info(f"Model trained successfully")
            self.logger.info(f"MAE: {metrics['mae']:.2f}s, RMSE: {metrics['rmse']:.2f}s")
            self.logger.info(f"R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to train model: {e}")
            raise

    def predict_arrival_time(self, current_state: Dict[str, Any]) -> float:
        """Predict arrival time for current bus state"""

        if self.model is None:
            raise ValueError("Model not trained yet")

        try:
            # Convert state to feature vector
            features = self._state_to_features(current_state)

            # Make prediction
            prediction = self.model.predict([features])[0]

            # Ensure reasonable bounds
            prediction = max(10, min(1800, prediction))  # 10s to 30min

            return prediction

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return 300  # Default 5-minute prediction

    def _state_to_features(self, state: Dict[str, Any]) -> List[float]:
        """Convert bus state to feature vector"""

        # Extract temporal features
        dt = datetime.fromtimestamp(state.get('timestamp', time.time()))

        features = {
            'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            'minute_sin': np.sin(2 * np.pi * dt.minute / 60),
            'minute_cos': np.cos(2 * np.pi * dt.minute / 60),
            'day_sin': np.sin(2 * np.pi * dt.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * dt.weekday() / 7),
            'is_peak_hour': int(dt.hour in [7, 8, 9, 17, 18, 19]),
            'is_weekend': int(dt.weekday() >= 5),
            'headway_backward': state.get('headway_backward', 600),
            'headway_forward': state.get('headway_forward', 600),
            'speed_kmh': state.get('speed_kmh', 25),
            'avg_speed_5min': state.get('avg_speed_5min', 25)
        }

        # Add encoded categorical features
        for col, encoder in self.label_encoders.items():
            value = state.get(col, 'unknown')
            try:
                encoded = encoder.transform([value])[0]
            except ValueError:
                encoded = 0  # Unknown category
            features[f'{col}_encoded'] = encoded

        # Convert to ordered list
        feature_vector = [features.get(col, 0) for col in self.feature_columns]

        return feature_vector

    def get_feature_importance(self) -> Dict[str, float]:
        """Get model feature importance"""

        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_columns, importance))

        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, filepath: str):
        """Save trained model to file"""

        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'model_params': self.model_params
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""

        try:
            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.label_encoders = model_data['label_encoders']
            self.model_params = model_data.get('model_params', {})

            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

# Example usage
if __name__ == "__main__":

    predictor = XGBoostArrivalPredictor()
