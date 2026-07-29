import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from WattPredictor.config.config import get_config
from WattPredictor.components.features.ingestion import Ingestion
from WattPredictor.components.features.validation import Validation
from WattPredictor.components.features.engineering import Engineering
from WattPredictor.utils.logging import logger


class FeaturePipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("STEP 1: Data Ingestion")
            logger.info("=" * 60)
            ingestor = Ingestion(config=self.config)
            ingestor.download()
            logger.info("Data ingestion completed")
            
            logger.info("=" * 60)
            logger.info("STEP 2: Data Validation")
            logger.info("=" * 60)
            validator = Validation(config=self.config)
            validator.validator()
            logger.info("Data validation completed")
            
            logger.info("=" * 60)
            logger.info("STEP 3: Feature Engineering")
            logger.info("=" * 60)
            transformer = Engineering(config=self.config)
            transformer.transform()
            logger.info("Feature engineering completed")
            
            logger.info("=" * 60)
            logger.info("FEATURE PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Feature pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
