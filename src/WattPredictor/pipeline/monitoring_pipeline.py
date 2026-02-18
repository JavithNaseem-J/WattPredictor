import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from WattPredictor.config.config import get_config
from WattPredictor.components.monitor.monitoring import Monitoring
from WattPredictor.entity.config_entity import MonitoringConfig
from WattPredictor.utils.logging import logger


class MonitoringPipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("PREDICTION MONITORING PIPELINE")
            logger.info("=" * 60)
            
            monitoring_config = MonitoringConfig(
                monitoring_df=self.config.artifacts_dir / "monitoring" / "monitoring_df.csv"
            )
            
            monitor = Monitoring(config=monitoring_config)
            monitoring_df = monitor.predictions_and_actuals()
            
            logger.info("=" * 60)
            logger.info("MONITORING COMPLETED")
            logger.info(f"Records compared: {len(monitoring_df)}")
            logger.info(f"Output: {monitoring_config.monitoring_df}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Monitoring pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    pipeline.run()
