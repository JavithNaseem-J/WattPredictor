from WattPredictor.config.data_config import ConfigurationManager
from WattPredictor.config.feature_config import FeatureConfigurationManager
from WattPredictor.components.data_drift import DriftDetector

class DataDriftPipeline:
    def __init__(self):
        pass
    def run(self):
        config = ConfigurationManager()
        drift_config = config.get_data_drift_config()
        feature_store_config = config.get_data_drift_config()
        drift_detector = DriftDetector(config=drift_config, feature_store_config=feature_store_config)
        drift_detected, report_dict = drift_detector.Detect()    