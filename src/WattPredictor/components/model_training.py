import os
import sys
import mlflow
import optuna
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from WattPredictor import logger
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from WattPredictor.entity.config_entity import ModelTrainerConfig
from WattPredictor.utils.helpers import create_directories, save_bin, load_json
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

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

    def train(self):
        train_x = np.load(self.config.x_transform, allow_pickle=True)
        train_y = np.load(self.config.y_transform, allow_pickle=True).squeeze()



        best_overall = {"model_name": None, "score": float("inf"), "params": None}

        for model_name, model_info in self.models.items():
            logger.info(f"Starting Optuna HPO for {model_name}")

            def objective(trial):
                params = model_info["search_space"](trial)
                model = model_info["class"](**params)

                # Train/val split for early stopping
                x_train, x_val, y_train, y_val = train_test_split(
                    train_x, train_y, test_size=0.2, shuffle=False
                )

                if model_name == "XGBoost":
                    model.set_params(early_stopping_rounds=0, eval_metric="rmse")
                    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                elif model_name == "LightGBM":
                    model.set_params(early_stopping_rounds=0)
                    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="rmse")

                preds = model.predict(x_val)
                rmse = root_mean_squared_error(y_val, preds)
                return rmse

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.config.n_trials)

            best_params = study.best_params
            logger.info(f"Best params for {model_name}: {best_params}")

            model = model_info["class"](**best_params)
            kf = KFold(n_splits=self.config.cv_folds, shuffle=False)
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

        # Log final model
        with mlflow.start_run(run_name=f"{best_overall['model_name']}_final"):
            mlflow.log_params(final_params)
            mlflow.log_metric("cv_rmse", best_overall["score"])
            mlflow.set_tag("stage", "final")

        logger.info(f"Best model: {best_overall}")
        return best_overall
