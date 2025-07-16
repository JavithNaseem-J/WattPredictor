from WattPredictor.config.model_config import ModelConfigurationManager
from WattPredictor.components.training.trainer import Trainer

class TrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        model_config = ModelConfigurationManager()
        trainer_config = model_config.get_model_trainer_config()
        trainer = Trainer(config=trainer_config)
        best_model_info = trainer.train()
        return best_model_info
