import yaml
import json
import joblib
from pathlib import Path
from box import ConfigBox
from WattPredictor.utils.logging import logger


def read_yaml(file_path: str) -> ConfigBox:
    try:
        with open(file_path, encoding="utf8") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {file_path} loaded successfully")
            return ConfigBox(config)
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {str(e)}")
        raise


def create_directories(dir_paths: list):
    for dir_path in dir_paths:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"created directory at: {dir_path}")


def save_json(file_path: str, data: dict):
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"json file saved at: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise


def load_json(file_path: str) -> dict:
    try:
        with open(file_path, encoding="utf8") as json_file:
            data = json.load(json_file)
        logger.info(f"json file loaded succesfully from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise


def save_bin(obj, file_path: str):
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Binary file saved at: {file_path}")
    except Exception as e:
        logger.error(f"Error saving binary file {file_path}: {str(e)}")
        raise


def load_bin(file_path: str):
    try:
        obj = joblib.load(file_path)
        logger.info(f"Binary file loaded from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading binary file {file_path}: {str(e)}")
        raise
