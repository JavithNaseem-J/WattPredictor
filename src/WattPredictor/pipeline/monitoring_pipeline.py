from WattPredictor.config.config_manager import ConfigManager
from WattPredictor.components.monitor.monitoring import Monitoring
from WattPredictor.components.monitor.drift import Drift


class MonitoringPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigManager()

        monitor_config = config.get_monitoring_config()
        monitor = Monitoring(config=monitor_config)
        monitor.predictions_and_actuals()

        drift_config = config.get_drift_config()
        drift = Drift(config=drift_config)
        drift.Detect()


if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    pipeline.run()
