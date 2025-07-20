from datetime import datetime, timedelta
from WattPredictor.config.inference_config import InferenceConfigurationManager
from WattPredictor.components.monitor.monitoring import Monitoring
from WattPredictor.components.monitor.drift import Drift
from WattPredictor.utils.exception import CustomException


class MonitoringPipeline:
    def __init__(self):
        pass

    def run(self):
        config = InferenceConfigurationManager()


        monitor_config = config.get_data_monitoring_config()
        monitor = Monitoring(config=monitor_config)
        monitor.predictions_and_actuals()

        drift_config = config.get_data_drift_config()
        drift = Drift(config=drift_config)
        drift.Detect()

