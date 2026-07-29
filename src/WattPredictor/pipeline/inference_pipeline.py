import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from WattPredictor.config.config import get_config
from WattPredictor.components.inference.predictor import Predictor
from WattPredictor.utils.logging import logger


class InferencePipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("INFERENCE PIPELINE")
            logger.info("=" * 60)
            
            predictor = Predictor(config=self.config)
            predictor.predict()
            
            logger.info("=" * 60)
            logger.info("INFERENCE COMPLETED SUCCESSFULLY")
            logger.info(f"Predictions saved to: {self.config.predictions_path}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = InferencePipeline()
    pipeline.run()
