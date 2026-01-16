import optuna
import joblib
import pandas as pd
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from datetime import datetime, timedelta
from WattPredictor.utils.helpers import create_directories, save_bin
from WattPredictor.utils.ts_generator import features_and_target, get_pipeline, average_demand_last_4_weeks
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.entity.config_entity import TrainerConfig
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.logging import logger



class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.feature_store = feature_store_instance()
        self.models = {
            "XGBoost": {
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                }
            },
            "LightGBM": {
                "search_space": lambda trial: {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                }
            }
        }

    def load_training_data(self):
        df, _ = self.feature_store.get_training_data("elec_wx_features_view")
        
        df = df[['date', 'demand', 'sub_region_code', 'temperature_2m', 
                 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']]
        df.sort_values("date", inplace=True)
        return df

    def train(self):
        logger.info("Starting model training process")
        df = self.load_training_data()
        if df.empty:
            raise CustomException("Loaded DataFrame is empty", None)
        
        cutoff_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        train_df, test_df = df[df['date'] < cutoff_date], df[df['date'] >= cutoff_date]
        if train_df.empty or test_df.empty:
            raise CustomException("Train or Test DataFrame is empty after applying cutoff_date", None)

        train_x, train_y = features_and_target(train_df, self.config.input_seq_len, self.config.step_size)
        train_x.drop(columns=["date"], errors="ignore", inplace=True)

        non_numeric_cols = train_x.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
        if not non_numeric_cols.empty:
            raise CustomException(f"Non-numeric columns found in train_x: {non_numeric_cols}", None)

        train_x_transformed = train_x.copy()
        train_x_transformed = average_demand_last_4_weeks(train_x_transformed)

        best_overall = {"model_name": None, "score": float("inf"), "params": None}

        for model_name, model_info in self.models.items():
            logger.info(f"Optimizing hyperparameters for {model_name}")
            def objective(trial):
                params = model_info["search_space"](trial)
                pipeline = get_pipeline(model_type=model_name, **params)
                x_tr, x_val, y_tr, y_val = train_test_split(train_x, train_y, test_size=0.2, shuffle=False)
                pipeline.fit(x_tr, y_tr)
                preds = pipeline.predict(x_val)
                return mean_squared_error(y_val, preds)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.config.n_trials)

            best_params = study.best_params
            pipeline = get_pipeline(model_type=model_name, **best_params)
            score = -cross_val_score(pipeline, train_x, train_y, cv=KFold(n_splits=self.config.cv_folds), 
                                     scoring="neg_root_mean_squared_error").mean()

            logger.info(f"{model_name} RMSE: {score:.4f}")
            if score < best_overall["score"]:
                best_overall.update({
                    "model_name": model_name,
                    "score": score,
                    "params": best_params
                })

        final_pipeline = get_pipeline(model_type=best_overall["model_name"], **best_overall["params"])
        final_pipeline.fit(train_x, train_y)

        model_path = self.config.root_dir / self.config.model_name
        create_directories([model_path.parent])
        save_bin(final_pipeline, model_path)

        input_schema = Schema(train_x_transformed.head(10))
        output_schema = Schema(pd.DataFrame(train_y))
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

        model_registry = self.feature_store.project.get_model_registry()
        training_timestamp = datetime.now().isoformat()
        hops_model = model_registry.python.create_model(
            name=f"wattpredictor_{best_overall['model_name'].lower()}",
            metrics={"rmse": best_overall["score"]},
            description=f"Model trained on data up to {cutoff_date}. Training timestamp: {training_timestamp}",
            input_example=train_x_transformed.head(10),
            model_schema=model_schema
        )
        hops_model.save(model_path.as_posix())
        logger.info(f"Best model registered: {best_overall['model_name']} v{hops_model.version} with RMSE {best_overall['score']:.4f}")
        return best_overall