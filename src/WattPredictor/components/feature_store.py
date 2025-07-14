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


    def create_feature_group(self, name, df, primary_key, event_time, description, online_enabled=True, version=1):
        try:
            try:
                fg = self.feature_store.get_feature_group(name=name, version=version)
                logger.info(f"Feature Group '{name}' v{version} exists. Deleting it.")
                fg.delete()
            except Exception:
                logger.info(f"Feature Group '{name}' v{version} does not exist. Will create a new one.")

            # Create a new feature group
            logger.info(f"Creating Feature Group '{name}' v{version}.")
            fg = self.feature_store.get_or_create_feature_group(
                name=name,
                version=version,
                primary_key=primary_key,
                event_time=event_time,
                description=description,
                online_enabled=online_enabled
            )

            fg.save(df)
            logger.info(f"Feature Group '{name}' v{version} created and data inserted.")

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


    def get_training_data(self, feature_view_name: str):
        try:
            fv = self.feature_store.get_feature_view(name=feature_view_name, version=1)
            X, y = fv.training_data()
            logger.info(f"Retrieved training data from Feature View '{feature_view_name}'")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def get_online_features(self, feature_view_name, key_dict: dict, version=1):
        try:
            fv = self.feature_store.get_feature_view(name=feature_view_name, version=version)
            if fv is None:
                logger.error(f"[Online Fetch] Feature View '{feature_view_name}' v{version} not found.")
                raise CustomException(f"Feature View '{feature_view_name}' v{version} is None", sys)

            expected_primary_keys = ["date_str", "sub_region_code"]
            
            key_values = [key_dict[key] for key in expected_primary_keys]
            
            try:
                result = fv.get_feature_vector(key_dict)
                logger.info(f"[Online Fetch] Fetched online features using get_feature_vector for {key_dict}: {result}")
                return result
            except Exception as vector_error:
                logger.warning(f"get_feature_vector failed: {vector_error}, trying get_serving_vector")
                
                result = fv.get_serving_vector(key_values).to_dict()
                logger.info(f"[Online Fetch] Fetched online features using get_serving_vector for {key_dict}: {result}")
                return result

        except Exception as e:
            logger.error(f"[Online Fetch] Failed to fetch online features for {feature_view_name} with key {key_dict}")
            raise CustomException(e, sys)