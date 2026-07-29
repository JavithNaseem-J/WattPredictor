import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from WattPredictor.config.config import get_config
from WattPredictor.components.monitor.monitoring import Monitoring
from WattPredictor.utils.logging import logger


class MonitoringPipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("PREDICTION MONITORING PIPELINE")
            logger.info("=" * 60)
            
            monitor = Monitoring(config=self.config)
            monitoring_df = monitor.predictions_and_actuals()
            
            logger.info("=" * 60)
            logger.info("MONITORING COMPLETED")
            logger.info(f"Records compared: {len(monitoring_df)}")
            logger.info(f"Output: {self.config.monitoring_df_path}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Monitoring pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    pipeline.run()
