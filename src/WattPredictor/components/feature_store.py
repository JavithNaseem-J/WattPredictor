import hopsworks
import pandas as pd
import sys
import os
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger

class FeatureStore:
    def __init__(self, config):
        try:
            self.config = config
            self.connect()
        except Exception as e:
            raise CustomException(e, sys)

    def connect(self):
        try:
            self.project = hopsworks.login(
                project=self.config.hopsworks_project_name,
                api_key_value=self.config.hopsworks_api_key
            )
            self.feature_store = self.project.get_feature_store()
            self.dataset_api = self.project.get_dataset_api()
            logger.info(f"Connected to Hopsworks Feature Store: {self.config.hopsworks_project_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def create_feature_group(self, name, df, primary_key, event_time, description):
        try:
            try:
                fg = self.feature_store.get_feature_group(name=name, version=1)
                logger.info(f"Feature Group '{name}' already exists. Inserting data instead.")
                fg.insert(df)
            except:
                logger.info(f"Feature Group '{name}' does not exist. Creating new one.")
                fg = self.feature_store.get_or_create_feature_group(
                    name=name,
                    version=1,
                    primary_key=primary_key,
                    event_time=event_time,
                    description=description,
                    online_enabled=False
                )
                fg.save(df)

            logger.info(f"Feature Group '{name}' created/updated successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def create_feature_view(self, name: str, feature_group_name: str, features: list):
        try:
            fg = self.feature_store.get_feature_group(name=feature_group_name, version=1)
            fv = self.feature_store.get_or_create_feature_view(
                name=name,
                version=1,
                query=fg.select(features),
                description=f"Feature View for {name}"
            )
            logger.info(f"Feature View '{name}' created successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def upload_file_safely(self, local_path: str, target_name: str):
        """
        Upload file to Hopsworks dataset storage.
        If it already exists, it will be overwritten.
        """
        try:
            self.dataset_api.upload(
                local_path,
                f"Resources/wattpredictor_artifacts/{target_name}",
                overwrite=True 
            )
            logger.info(f"Uploaded file to Feature Store: {target_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def delete_file(self, target_name: str):

        try:
            full_path = f"Resources/wattpredictor_artifacts/{target_name}"
            self.dataset_api.delete(full_path)
            logger.warning(f"Deleted file from Feature Store: {target_name}")
        except Exception as e:
            logger.warning(f"File not found or already deleted: {target_name}")
            # Not raising exception here to allow safe cleanup

    def get_training_data(self, feature_view_name: str):
        try:
            fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
            X, y = fv.training_data()
            logger.info(f"Retrieved training data from Feature View '{feature_view_name}'")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)