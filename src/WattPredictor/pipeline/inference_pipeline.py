import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from WattPredictor.config.config import get_config
from WattPredictor.components.inference.predictor import Predictor
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.utils.logging import logger


class InferencePipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("INFERENCE PIPELINE")
            logger.info("=" * 60)
            
            predictor_config = PredictionConfig(
                model_path=self.config.model_path,
                predictions_df=self.config.predictions_path
            )
            
            predictor = Predictor(config=predictor_config)
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
