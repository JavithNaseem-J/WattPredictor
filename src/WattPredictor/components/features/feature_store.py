import hopsworks
import pandas as pd
import sys
import os
import joblib
from pathlib import Path
from WattPredictor.utils.logging import logger
from WattPredictor.config.feature_config import FeatureStoreConfig
from WattPredictor.utils.exception import CustomException


class FeatureStore:
    def __init__(self, config):
        self.config = config
        self.project = hopsworks.login(
            project=config.hopsworks_project_name,
            api_key_value=config.hopsworks_api_key
        )
        self.feature_store = self.project.get_feature_store()
        self.dataset_api = self.project.get_dataset_api()
        logger.info(f"Connected to Hopsworks Feature Store: {config.hopsworks_project_name}")

    def create_feature_group(self, name, df, primary_key, event_time, description, online_enabled=True, version=2):
        try:
            # Check if feature group exists
            fg = self.feature_store.get_feature_group(name=name, version=version)
            fg.insert(df)
            logger.info(f"Feature Group '{name}' v{version} exists. Data inserted.")
            return fg
        except:
            fg = self.feature_store.create_feature_group(
                name=name,
                version=version,
                primary_key=primary_key,
                event_time=event_time,
                description=description,
                online_enabled=online_enabled
            )
            fg.save(df)
            logger.info(f"Feature Group '{name}' v{version} created successfully.")
            return fg

    def create_feature_view(self, name, feature_group_name, features, version=1):
        try:
            fv = self.feature_store.get_feature_view(name=name, version=version)
            if fv is not None:
                logger.info(f"Feature View '{name}' v{version} already exists.")
                return fv
            else:
                raise Exception("Feature view not found")
        except Exception:
            fg = self.feature_store.get_feature_group(name=feature_group_name, version=2)
            fv = self.feature_store.create_feature_view(
                name=name,
                version=version,
                query=fg.select(features),
                description=f"Feature View for {name}"
            )
            logger.info(f"Feature View '{name}' v{version} created successfully.")
            return fv

    def save_training_dataset(self, feature_view_name, version_description, output_format="csv"):
        fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
        if fv is None:
            raise Exception(f"Feature view '{feature_view_name}' not found or is None")
        
        td = fv.create_training_data(
            description=version_description,
            data_format=output_format,
            write_options={"wait_for_job": True}
        )
        logger.info(f"Training dataset created for Feature View '{feature_view_name}'.")
        return td

    def get_training_data(self, feature_view_name):
        fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
        X, y = fv.training_data()
        logger.info(f"Retrieved training data from Feature View '{feature_view_name}'.")
        return X, y
    

    def get_online_features(self, feature_view_name, key_dict, version=1):
        fv = self.feature_store.get_feature_view(name=feature_view_name, version=version)
        result = fv.get_feature_vector(key_dict)
        logger.info(f"Retrieved online features for '{feature_view_name}' with key {key_dict}.")
        return result

    def upload_file(self, local_path, target_name):
        self.dataset_api.upload(
            local_path,
            f"Resources/wattpredictor_artifacts/{target_name}",
            overwrite=True
        )
        logger.info(f"File uploaded to Feature Store: {target_name}")

    def load_model(self, model_name, model_version, model_filename="model.joblib"):
        model_registry = self.project.get_model_registry()
        model_meta = model_registry.get_model(name=model_name, version=model_version)
        model_dir = Path(model_meta.download())
        model_path = model_dir / model_filename
        model = joblib.load(model_path)
        logger.info(f"Model '{model_name}' v{model_version} loaded successfully.")
        return model
    
    def upload_file_safely(self, local_path: str, target_name: str):

        try:
            self.dataset_api.upload(
                local_path,
                f"Resources/wattpredictor_artifacts/{target_name}",
                overwrite=True 
            )
            logger.info(f"Uploaded file to Feature Store: {target_name}")
        except Exception as e:
            raise CustomException(e, sys)