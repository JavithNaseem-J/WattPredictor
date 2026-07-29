import pandas as pd
from datetime import timedelta
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
import mlflow
import mlflow.sklearn
from WattPredictor.config.config import WattPredictorConfig, get_config
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.ts_generator import features_and_target, get_pipeline
from WattPredictor.utils.logging import logger


class Trainer:
    def __init__(self, config: WattPredictorConfig = None):
        self.config = config or get_config()
        self.param_grids = {
            "XGBoost": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 7],
                "model__learning_rate": [0.05, 0.1],
            },
            "LightGBM": {
                "model__num_leaves": [50, 100],
                "model__learning_rate": [0.05, 0.1],
                "model__n_estimators": [100, 200],
            }
        }

    def load_training_data(self):
        logger.info(f"Loading training data from {self.config.preprocessed_data_path}")
        df = pd.read_csv(self.config.preprocessed_data_path)
        
        df = df[['date', 'demand', 'sub_region_code', 'temperature_2m', 
                 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']]
        df.sort_values("date", inplace=True)
        return df

    def train(self):
        logger.info("Starting model training process with MLflow tracking")
        df = self.load_training_data()
        
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        df_date = pd.to_datetime(df['date'])
        max_date = df_date.max()
        cutoff_date = max_date - timedelta(days=90)
        
        train_df = df[df_date < cutoff_date]
        test_df = df[df_date >= cutoff_date]
        
        if train_df.empty or test_df.empty:
            raise ValueError("Train or Test DataFrame is empty after dataset cutoff split")

        train_x, train_y = features_and_target(train_df, self.config.input_seq_len, self.config.step_size)
        train_x.drop(columns=["date"], errors="ignore", inplace=True)

        non_numeric_cols = train_x.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
        if not non_numeric_cols.empty:
            raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")

        mlflow.set_experiment("WattPredictor")

        best_overall = {"model_name": None, "score": float("inf"), "params": None}

        with mlflow.start_run(run_name="GridSearch_Tuning") as parent_run:
            for model_name, param_grid in self.param_grids.items():
                logger.info(f"Tuning {model_name}")
                
                grid_search = GridSearchCV(
                    estimator=get_pipeline(model_type=model_name),
                    param_grid=param_grid,
                    cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                
                grid_search.fit(train_x, train_y)
                best_score = -grid_search.best_score_
                
                logger.info(f"{model_name} RMSE: {best_score:.4f}")

                with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True):
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("cv_best_rmse", best_score)
                
                if best_score < best_overall["score"]:
                    best_overall.update({
                        "model_name": model_name,
                        "score": best_score,
                        "params": grid_search.best_params_,
                        "estimator": grid_search.best_estimator_
                    })

            mlflow.log_param("best_model_type", best_overall["model_name"])
            mlflow.log_metric("best_cv_rmse", best_overall["score"])
            mlflow.log_params(best_overall["params"])
            mlflow.sklearn.log_model(best_overall["estimator"], "best_model")

        final_pipeline = best_overall["estimator"]
        model_path = self.config.model_path
        create_directories([model_path.parent])
        joblib.dump(final_pipeline, model_path)
        
        logger.info(f"Best model: {best_overall['model_name']} with RMSE {best_overall['score']:.4f}")
        logger.info(f"Model saved to {model_path} and logged to MLflow")
        
        return best_overall