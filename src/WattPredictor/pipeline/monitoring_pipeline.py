from WattPredictor.config.data_config import DataConfigurationManager
from WattPredictor.config.model_config import ModelConfigurationManager
from WattPredictor.components.monitor.drift import Drift
from src.WattPredictor.components.training.evaluator import Evaluation

class MonitoringPipeline:
    def __init__(self):
        pass

    def run(self):
        data_config = DataConfigurationManager()
        model_config = ModelConfigurationManager

        drift_config = data_config.get_data_drift_config()
        drift_detector = Drift(config=drift_config)
        drift_detected, drift_report = drift_detector.Drift()

        evaluator_config = model_config.get_model_evaluation_config()
        evaluator = Evaluation(config=evaluator_config)
        evaluation_metrics = evaluator.evaluate()

        return {
            "drift_detected": drift_detected,
            "drift_report": drift_report,
            "evaluation_metrics": evaluation_metrics
        }
