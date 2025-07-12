import hopsworks
import pandas as pd
import sys
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger


class FeatureStore:
    def __init__(self, config):
        self.config = config
        self.connect()

    def connect(self):
        try:
            self.project = hopsworks.login(
                project=self.config.hopsworks_project_name,
                api_key_value=self.config.hopsworks_api_key
            )
            self.feature_store = self.project.get_feature_store()
            self.dataset_api = self.project.get_dataset_api()
            logger.info(f"Connected to Hopsworks Project: {self.config.hopsworks_project_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def create_feature_group(self, name, df, primary_key, event_time, description, online=False):
        try:
            try:
                fg = self.feature_store.get_feature_group(name=name, version=1)
                logger.info(f"Feature Group '{name}' exists. Inserting data.")
            except Exception:
                logger.info(f"Feature Group '{name}' not found. Creating it.")
                fg = self.feature_store.create_feature_group(
                    name=name,
                    version=1,
                    primary_key=primary_key,
                    event_time=event_time,
                    description=description,
                    online_enabled=online
                )
                logger.info(f"Feature Group '{name}' created.")

            fg.insert(df, write_options={"wait_for_job": True})
            logger.info(f"Inserted {len(df)} rows into Feature Group '{name}'.")
            return fg

        except Exception as e:
            logger.error(f"Error in Feature Group '{name}': {str(e)}")
            raise CustomException(e, sys)

    def create_feature_view(self, name, feature_group_name, features):
        try:
            try:
                existing_fv = self.feature_store.get_feature_view(name=name, version=1)
                existing_fv.delete()
                logger.info(f"Deleted existing Feature View '{name}' for recreation.")
            except Exception:
                logger.info(f"Creating new Feature View '{name}'.")

            fg = self.feature_store.get_feature_group(name=feature_group_name, version=1)
            query = fg.select(features)

            fv = self.feature_store.create_feature_view(
                name=name,
                version=1,
                query=query,
                description=f"Feature View for {feature_group_name}"
            )
            logger.info(f"Feature View '{name}' created successfully.")
            return fv

        except Exception as e:
            raise CustomException(e, sys)

    def save_training_dataset(self, feature_view_name, version_description, output_format="csv"):
        try:
            fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
            td = fv.create_training_data(
                description=version_description,
                data_format=output_format,
                write_options={"wait_for_job": True}
            )
            logger.info(f"Training dataset created for Feature View '{feature_view_name}'.")
            return td
        except Exception as e:
            raise CustomException(e, sys)

    def load_latest_training_dataset(self, feature_view_name):
        try:
            fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
            return fv.training_data()
        except Exception as e:
            raise CustomException(e, sys)

    def get_online_features(self, feature_view_name, key_dict: dict):
        try:
            fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
            return fv.get_online_features(key_dict)
        except Exception as e:
            raise CustomException(e, sys)

    def upload_file_safely(self, local_path: str, target_name: str):
        try:
            self.dataset_api.upload(
                local_path,
                f"Resources/wattpredictor_artifacts/{target_name}",
                overwrite=True
            )
            logger.info(f"Uploaded file: {target_name} to Hopsworks Dataset Storage.")
        except Exception as e:
            raise CustomException(e, sys)
