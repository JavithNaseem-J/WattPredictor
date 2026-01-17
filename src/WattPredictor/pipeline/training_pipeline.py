from WattPredictor.config.model_config import ModelConfigurationManager
from WattPredictor.components.training.trainer import Trainer
from WattPredictor.components.training.evaluator import Evaluation
from WattPredictor.utils.exception import CustomException



class TrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ModelConfigurationManager()

        trainer_config = config.get_model_trainer_config()
        trainer = Trainer(config=trainer_config)
        trainer.train()

        evaluation_config = config.get_model_evaluation_config()
        evaluator = Evaluation(config=evaluation_config)
        evaluator.evaluate()


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()

