import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from WattPredictor.config.config import get_config
from WattPredictor.components.training.trainer import Trainer
from WattPredictor.components.training.evaluator import Evaluation
from WattPredictor.components.monitor.drift import Drift
from WattPredictor.utils.logging import logger


class TrainingPipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("STEP 1: Model Training")
            logger.info("=" * 60)
            trainer = Trainer(config=self.config)
            best_model = trainer.train()
            logger.info(f"Best model: {best_model['model_name']} (RMSE: {best_model['score']:.2f} MW)")
            
            logger.info("=" * 60)
            logger.info("STEP 2: Model Evaluation")
            logger.info("=" * 60)
            evaluator = Evaluation(config=self.config)
            metrics = evaluator.evaluate()
            logger.info(f"RMSE: {metrics['rmse']:.2f} MW | MAPE: {metrics['mape']:.2f}%")
            
            logger.info("=" * 60)
            logger.info("STEP 3: Drift Detection (Evidently)")
            logger.info("=" * 60)
            drift = Drift(config=self.config)
            drift_detected, _ = drift.Detect()
            
            if drift_detected:
                logger.warning("DATA DRIFT DETECTED - Review drift report")
            else:
                logger.info("No significant data drift detected")
            
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Model: {best_model['model_name']}")
            logger.info(f"Validation RMSE: {metrics['rmse']:.2f} MW")
            logger.info(f"Validation MAPE: {metrics['mape']:.2f}%")
            logger.info(f"Drift detected: {drift_detected}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
