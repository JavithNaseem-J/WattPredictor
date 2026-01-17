import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from WattPredictor.utils.helpers import create_directories, save_bin
from WattPredictor.utils.ts_generator import features_and_target, get_pipeline, average_demand_last_4_weeks
from WattPredictor.entity.config_entity import TrainerConfig
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.logging import logger



class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.feature_store = None
        self.use_hopsworks = os.getenv("USE_HOPSWORKS", "false").lower() == "true"
        if self.use_hopsworks:
            try:
                from WattPredictor.utils.feature import feature_store_instance
                self.feature_store = feature_store_instance()
            except Exception as e:
                logger.warning(f"Hopsworks connection failed: {e}. Using local data.")
                self.use_hopsworks = False
        # Parameter grids for GridSearchCV - prefix with 'model__' for Pipeline
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
        if self.use_hopsworks and self.feature_store:
            df, _ = self.feature_store.get_training_data("elec_wx_features_view")
        else:
            # Load from local preprocessed file
            logger.info("Loading training data from local file")
            df = pd.read_csv(self.config.data_path)
        
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

        best_overall = {"model_name": None, "score": float("inf"), "params": None}

        # Simple grid search for each model (Pipeline includes feature transformation)
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
            
            if best_score < best_overall["score"]:
                best_overall.update({
                    "model_name": model_name,
                    "score": best_score,
                    "params": grid_search.best_params_,
                    "estimator": grid_search.best_estimator_
                })

        final_pipeline = best_overall["estimator"]

        model_path = self.config.root_dir / self.config.model_name
        create_directories([model_path.parent])
        save_bin(final_pipeline, model_path)
        logger.info(f"Model saved to {model_path}")

        # Register model to Hopsworks if enabled
        if self.use_hopsworks and self.feature_store:
            try:
                from hsml.schema import Schema
                from hsml.model_schema import ModelSchema
                
                input_schema = Schema(train_x.head(10))
                output_schema = Schema(pd.DataFrame(train_y))
                model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

                model_registry = self.feature_store.project.get_model_registry()
                training_timestamp = datetime.now().isoformat()
                hops_model = model_registry.python.create_model(
                    name=f"wattpredictor_{best_overall['model_name'].lower()}",
                    metrics={"rmse": best_overall["score"]},
                    description=f"Model trained on data up to {cutoff_date}. Training timestamp: {training_timestamp}",
                    input_example=train_x.head(10),
                    model_schema=model_schema
                )
                hops_model.save(model_path.as_posix())
                logger.info(f"Model registered to Hopsworks: {best_overall['model_name']} v{hops_model.version}")
            except Exception as e:
                logger.warning(f"Failed to register model to Hopsworks: {e}")

        logger.info(f"Best model: {best_overall['model_name']} with RMSE {best_overall['score']:.4f}")
        
        return best_overall