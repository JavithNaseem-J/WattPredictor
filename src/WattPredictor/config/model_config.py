from pathlib import Path
from WattPredictor.entity.config_entity import TrainerConfig,EvaluationConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants.paths import *


class ModelConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> TrainerConfig:
        config = self.config.trainer
        params = self.params.training

        create_directories([config.root_dir])

        model_trainer_config = TrainerConfig(
            root_dir=Path(config.root_dir),
            input_seq_len= params.input_seq_len,
            step_size = params.step_size,
            cv_folds=params.cv_folds,
            model_name = Path(config.model_name),
            data_path = Path(config.data_path)
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        params = self.params.training

        create_directories([config.root_dir])
        
        model_evaluation_config =  EvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            input_seq_len= params.input_seq_len,
            step_size = params.step_size,
            img_path=Path(config.img_path),
            metrics_path=Path(config.metrics_path)
        )

        return model_evaluation_config