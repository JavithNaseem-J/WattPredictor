import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from WattPredictor.config.config import get_config
from WattPredictor.components.training.trainer import Trainer
from WattPredictor.components.training.evaluator import Evaluation
from WattPredictor.components.monitor.drift import Drift
from WattPredictor.entity.config_entity import TrainerConfig, EvaluationConfig, DriftConfig
from WattPredictor.utils.logging import logger


class TrainingPipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            # Step 1: Train model
            logger.info("=" * 60)
            logger.info("STEP 1: Model Training")
            logger.info("=" * 60)
            trainer_config = TrainerConfig(
                root_dir=self.config.artifacts_dir / "trainer",
                input_seq_len=self.config.input_seq_len,
                step_size=self.config.step_size,
                cv_folds=self.config.cv_folds,
                model_name=Path("model.joblib"),
                data_path=self.config.preprocessed_data_path
            )
            trainer = Trainer(config=trainer_config)
            best_model = trainer.train()
            logger.info(f"Best model: {best_model['model_name']} (RMSE: {best_model['score']:.2f} MW)")
            
            # Step 2: Evaluate model
            logger.info("=" * 60)
            logger.info("STEP 2: Model Evaluation")
            logger.info("=" * 60)
            eval_config = EvaluationConfig(
                root_dir=self.config.artifacts_dir / "evaluation",
                model_path=self.config.model_path,
                input_seq_len=self.config.input_seq_len,
                step_size=self.config.step_size,
                img_path=self.config.artifacts_dir / "evaluation" / "pred_vs_actual.png",
                metrics_path=self.config.metrics_path
            )
            evaluator = Evaluation(config=eval_config)
            metrics = evaluator.evaluate()
            logger.info(f"RMSE: {metrics['rmse']:.2f} MW | MAPE: {metrics['mape']:.2f}%")
            
            # Step 3: Drift detection
            logger.info("=" * 60)
            logger.info("STEP 3: Drift Detection (Evidently)")
            logger.info("=" * 60)
            drift_config = DriftConfig(
                report_dir=self.config.drift_report_html.parent
            )
            drift = Drift(config=drift_config)
            drift_detected, _ = drift.Detect()
            
            if drift_detected:
                logger.warning("DATA DRIFT DETECTED - Review drift report")
            else:
                logger.info("No significant data drift detected")
            
            # Summary
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
