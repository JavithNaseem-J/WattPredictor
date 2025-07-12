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
        params = self.params.training

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            input_seq_len= params.input_seq_len,
            step_size = params.step_size,
            n_trials=params.n_trials,
            cutoff_date = params.cutoff_date,
            model_name = Path(config.model_name)
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.training

        model_evaluation_config =  ModelEvaluationConfig(
            model_path=Path(config.model_path),
            cutoff_date=params.cutoff_date,
            input_seq_len= params.input_seq_len,
            step_size = params.step_size,
            metrics_path=Path(config.metrics_path)
        )

        return model_evaluation_config