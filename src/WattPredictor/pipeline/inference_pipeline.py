from WattPredictor.config.config_manager import ConfigManager
from WattPredictor.components.inference.predictor import Predictor


class InferencePipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigManager()
        predictor_config = config.get_prediction_config()
        predictor = Predictor(config=predictor_config)
        predictor.predict(save_to_store=True)


if __name__ == "__main__":
    pipeline = InferencePipeline()
    pipeline.run()
