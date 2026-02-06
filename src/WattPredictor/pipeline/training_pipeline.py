from WattPredictor.config.config_manager import ConfigManager
from WattPredictor.components.training.trainer import Trainer
from WattPredictor.components.training.evaluator import Evaluation


class TrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigManager()

        trainer_config = config.get_trainer_config()
        trainer = Trainer(config=trainer_config)
        trainer.train()

        evaluation_config = config.get_evaluation_config()
        evaluator = Evaluation(config=evaluation_config)
        evaluator.evaluate()


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
