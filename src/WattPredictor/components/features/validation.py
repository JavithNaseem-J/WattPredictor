import os
import json
import pandas as pd
from WattPredictor.utils.logging import logger
from WattPredictor.entity.config_entity import ValidationConfig
from WattPredictor.config.data_config import DataConfigurationManager
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.exception import CustomException

class Validation:
    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_data_types(self, data: pd.DataFrame, schema: dict):

        type_mapping = {
            'int': ['int64', 'int32'],
            'float': ['float64', 'float32'],
            'object': ['object'],
            'str': ['object'], 
        }

        for col, expected_type in schema.items():
            if col not in data.columns:
                continue 
                
            actual_dtype = str(data[col].dtype)
            allowed_dtypes = type_mapping.get(expected_type, [expected_type])

            if actual_dtype not in allowed_dtypes:
                logger.error(f"Column '{col}': Expected type '{expected_type}', got '{actual_dtype}'")
                return False
        return True

    def validate_column_presence(self, data: pd.DataFrame, schema: dict):
        all_cols = list(data.columns)
        expected_cols = set(schema.keys())
        missing_cols = expected_cols - set(all_cols)

        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        return True
    
    def check_missing_values(self, data: pd.DataFrame, threshold: float = 0.05) -> bool:
        missing_percent = data[['date_str', 'date', 'subba', 'value', 'temperature_2m']].isnull().mean()
        flagged = missing_percent[missing_percent > threshold]
        if not flagged.empty:
            logger.error(f"Columns exceeding {threshold*100}% missing:\n{flagged}")
            return False
        return True


    def validator(self):
        data = pd.read_csv(self.config.data_file)
        schema = self.config.all_schema

        logger.info(f"Starting validation for data with shape: {data.shape}")
            
        validation_results = {}
            
        validation_results = {
            'column_presence': self.validate_column_presence(data, schema),
            'data_types': self.validate_data_types(data, schema),
            'missing_values': self.check_missing_values(data),
        }
            
        is_valid = all(validation_results.values())

        create_directories([os.path.dirname(self.config.status_file)])

        for check, result in validation_results.items():
            logger.info(f"{check}: {'PASSED' if result else 'FAILED'}")

        logger.info(f"Overall validation status: {'PASSED' if is_valid else 'FAILED'}")

        with open(self.config.status_file, 'w') as f:
            json.dump({"validation_status": is_valid}, f, indent=4)

        return is_valid