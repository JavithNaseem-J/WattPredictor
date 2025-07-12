import os
import sys
import optuna
import joblib
import pandas as pd
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from WattPredictor.utils.helpers import *
from WattPredictor.utils.ts_generator import features_and_target
from WattPredictor.config.feature_config import FeatureStoreConfig
from WattPredictor.config.model_config import ModelTrainerConfig
from WattPredictor.components.feature_store import FeatureStore
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, feature_store_config: FeatureStoreConfig):
        self.config = config
        self.feature_store = FeatureStore(feature_store_config)

        self.models = {
            "XGBoost": {
                "class": XGBRegressor,
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                }
            },
            "LightGBM": {
                "class": LGBMRegressor,
                "search_space": lambda trial: {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                }
            }
        }

    def load_training_data(self):
        try:
            df, _ = self.feature_store.load_latest_training_dataset("elec_wx_features_view")
            df = df[['date', 'demand', 'sub_region_code', 'temperature_2m']]
            df.sort_values("date", inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def train(self):
        try:
            df = self.load_training_data()
            train_df = df[df['date'] < self.config.cutoff_date]
            test_df = df[df['date'] >= self.config.cutoff_date]

            train_x, train_y = features_and_target(train_df, self.config.input_seq_len, self.config.step_size)
            train_x.drop(columns=["date"], errors="ignore", inplace=True)

            best_overall = {"model_name": None, "score": float("inf"), "params": None}

            for model_name, model_info in self.models.items():
                logger.info(f"Running Optuna HPO for {model_name}")

                def objective(trial):
                    params = model_info["search_space"](trial)
                    model = model_info["class"](**params)
                    x_tr, x_val, y_tr, y_val = train_test_split(train_x, train_y, test_size=0.2, shuffle=False)
                    model.fit(x_tr, y_tr)
                    preds = model.predict(x_val)
                    return mean_squared_error(y_val, preds)

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=self.config.n_trials)

                best_params = study.best_params
                model = model_info["class"](**best_params)
                score = -cross_val_score(model, train_x, train_y, cv=KFold(n_splits=5), scoring="neg_root_mean_squared_error").mean()

                if score < best_overall["score"]:
                    best_overall.update({
                        "model_name": model_name,
                        "score": score,
                        "params": best_params
                    })

            final_model_class = self.models[best_overall["model_name"]]["class"]
            final_model = final_model_class(**best_overall["params"])
            final_model.fit(train_x, train_y)

            model_path = Path(self.config.root_dir) / self.config.model_name
            create_directories([model_path.parent])
            save_bin(final_model, model_path)

            # Create schema from training data
            input_schema = Schema(train_x)
            output_schema = Schema(pd.DataFrame(train_y))
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

            model_registry = self.feature_store.project.get_model_registry()
            hops_model = model_registry.python.create_model(
                name="wattpredictor_" + best_overall["model_name"].lower(),
                input_example=train_x.head(2),
                model_schema=model_schema,
                description="Best model trained on electricity demand"
            )
            hops_model.save(model_path.as_posix())

            logger.info(f"Best model registered: {best_overall}")
            return best_overall

        except Exception as e:
            raise CustomException(e, sys)