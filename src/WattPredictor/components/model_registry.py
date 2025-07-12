import sys
import joblib
from pathlib import Path
from WattPredictor.utils.helpers import create_directories, save_bin
from WattPredictor.components.feature_store import FeatureStore
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger


class ModelRegistry:
    def __init__(self, feature_store_config):
        self.feature_store = FeatureStore(feature_store_config)
        self.project = self.feature_store.project
        self.registry = self.project.get_model_registry()

    def register_model(self, model_object, model_name: str, model_path: str, input_example, model_schema):
        try:
            # Save locally if needed
            create_directories([Path(model_path).parent])
            save_bin(model_object, model_path)

            # Register to Hopsworks
            model = self.registry.python.create_model(
                name=model_name,
                model_path=model_path,
                input_example=input_example,
                model_schema=model_schema,
                description=f"{model_name} trained model"
            )
            model.save()
            logger.info(f"✅ Model '{model_name}' registered successfully to Hopsworks.")

        except Exception as e:
            raise CustomException(e, sys)

    def load_latest_model(self, model_name: str):
        try:
            model = self.registry.get_model(model_name, version=1)
            model.download()
            model_instance = model.load()
            logger.info(f"✅ Model '{model_name}' loaded successfully from Hopsworks.")
            return model_instance
        except Exception as e:
            raise CustomException(e, sys)
