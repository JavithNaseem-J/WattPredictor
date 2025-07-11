from pathlib import Path
from WattPredictor.entity.config_entity import ModelTrainerConfig, ModelEvaluationConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants import *


class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer
        trans = self.params.transformation

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_features= Path(config.train_features),
            test_features= Path(config.test_features),
            x_transform= Path(config.x_transform),
            y_transform= Path(config.y_transform),
            model_name=config.model_name,
            input_seq_len= trans.input_seq_len,
            step_size = trans.step_size,
            n_trials=params.n_trials
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        model_evaluation_config =  ModelEvaluationConfig(
            model_path=Path(config.model_path),
            x_transform=Path(config.X_transform),
            y_transform=Path(config.y_transform),
            metrics_path=Path(config.metrics_path)
        )

        return model_evaluation_config