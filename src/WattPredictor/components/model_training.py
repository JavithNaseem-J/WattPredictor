import os
import sys
import mlflow
import optuna
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from WattPredictor.utils.helpers import *
from WattPredictor.config.feature_config import FeatureStoreConfig
from WattPredictor.config.model_config import ModelTrainerConfig
from WattPredictor.components.feature_store import FeatureStore
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger
from sklearn.preprocessing import StandardScaler
import xgboost
import lightgbm


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, feature_store_config):
        self.config = config
        self.feature_store_config = feature_store_config
        self.feature_store = FeatureStore(feature_store_config)

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Electricity Demand Prediction")

        logger.info("MLflow tracking setup complete.")


        self.models = {
            "XGBoost": {
                "class": XGBRegressor,
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                },
                "mlflow_module": mlflow.xgboost,
            },
            "LightGBM": {
                "class": LGBMRegressor,
                "search_space": lambda trial: {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                },
                "mlflow_module": mlflow.lightgbm,
            },
        }

    def load_inputs(self):
        try:
            self.feature_store.dataset_api.download("Resources/wattpredictor_artifacts/train_df.csv/train_features.csv", overwrite=True)
            self.feature_store.dataset_api.download("Resources/wattpredictor_artifacts/test_df.csv/test_features.csv", overwrite=True)
            train_df = pd.read_csv(self.config.train_features)
            test_df = pd.read_csv(self.config.test_features)

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)

    def get_cutoff_indices(self, df: pd.DataFrame, input_seq_len: int, step_size: int):
        stop = len(df) - input_seq_len - 1
        return [(i, i + input_seq_len, i + input_seq_len + 1) for i in range(0, stop, step_size)]
        
    def generate_ts_features_and_target(self, ts_data: pd.DataFrame):
            
            assert set(['date', 'demand', 'sub_region_code', 'temperature_2m']).issubset(ts_data.columns)

            region_codes = ts_data['sub_region_code'].unique()
            features = pd.DataFrame()
            targets = pd.DataFrame()

            input_seq_len = self.config.input_seq_len
            step_size = self.config.step_size

            for code in tqdm(region_codes, desc="Transforming TS Data"):
                ts_one = ts_data[ts_data['sub_region_code'] == code].sort_values(by='date')
                indices = self.get_cutoff_indices(ts_one, input_seq_len, step_size)

                x = np.zeros((len(indices), input_seq_len), dtype=np.float64)
                y = np.zeros((len(indices)), dtype=np.float64)
                date_hours, temps = [], []

                for i, (start, mid, end) in enumerate(indices):
                    x[i, :] = ts_one.iloc[start:mid]['demand'].values
                    y[i] = ts_one.iloc[mid]['demand']
                    date_hours.append(ts_one.iloc[mid]['date'])
                    temps.append(ts_one.iloc[mid]['temperature_2m'])

                features_one = pd.DataFrame(
                    x,
                    columns=[f'demand_prev_{i+1}_hr' for i in reversed(range(input_seq_len))]
                )
                features_one['date'] = date_hours
                features_one['sub_region_code'] = code
                features_one['temperature_2m'] = temps

                targets_one = pd.DataFrame(y, columns=['target_demand_next_hour'])

                features = pd.concat([features, features_one], ignore_index=True)
                targets = pd.concat([targets, targets_one], ignore_index=True)

            return features, targets['target_demand_next_hour']

    def train(self):
        train_df, test_df = self.load_inputs()
        train_x, train_y = self.generate_ts_features_and_target(train_df)
        test_x, test_y = self.generate_ts_features_and_target(test_df)
        
        train_x = train_x.drop(columns=["date"], errors="ignore")
        test_x = test_x.drop(columns=["date"], errors="ignore")

        logger.info(f'shape of train_x:{train_x.shape}, train_y:{train_y.shape}, test_x:{test_x.shape}, test_y:{test_y.shape}')

        test_x.to_parquet(self.config.x_transform)
        test_y.to_frame().to_parquet(self.config.y_transform)


        self.feature_store.upload_file_safely(self.config.x_transform, os.path.basename(self.config.x_transform))
        self.feature_store.upload_file_safely(self.config.y_transform, os.path.basename(self.config.y_transform))

        best_overall = {"model_name": None, "score": float("inf"), "params": None}

        for model_name, model_info in self.models.items():
            logger.info(f"Starting Optuna HPO for {model_name}")

            def objective(trial):
                params = model_info["search_space"](trial)
                model = model_info["class"](**params)

                x_train, x_val, y_train, y_val = train_test_split(
                    train_x, train_y, test_size=0.2, shuffle=False
                )

                model.fit(x_train, y_train)
                preds = model.predict(x_val)
                rmse = root_mean_squared_error(y_val, preds)
                return rmse

            # Create and run the study for this model
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.config.n_trials)

            best_params = study.best_params
            logger.info(f"Best params for {model_name}: {best_params}")

            model = model_info["class"](**best_params)
            kf = KFold(n_splits=5, shuffle=False)
            scores = cross_val_score(model, train_x, train_y, cv=kf, scoring="neg_root_mean_squared_error")
            mean_score = -scores.mean()

            with mlflow.start_run(run_name=f"{model_name}_best"):
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_rmse", mean_score)
                mlflow.set_tag("model_name", model_name)

            if mean_score < best_overall["score"]:
                best_overall.update({
                    "model_name": model_name,
                    "score": mean_score,
                    "params": best_params
                })

        best_model_class = self.models[best_overall["model_name"]]["class"]
        final_params = best_overall["params"]
        best_model = best_model_class(**final_params)
        best_model.fit(train_x, train_y)

        model_path = Path(self.config.root_dir) / self.config.model_name
        create_directories([model_path.parent])
        save_bin(best_model, model_path)

        self.feature_store.upload_file_safely(model_path, "model.joblib")

        with mlflow.start_run(run_name=f"{best_overall['model_name']}_final"):
            mlflow.log_params(final_params)
            mlflow.log_metric("cv_rmse", best_overall["score"])
            mlflow.set_tag("stage", "final")

        logger.info(f"Best model: {best_overall}")
        return best_overall