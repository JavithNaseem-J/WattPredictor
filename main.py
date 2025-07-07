import os
import sys
import argparse
from WattPredictor import logger
from WattPredictor.pipelines.stage_01 import DataIngestionPipeline
from WattPredictor.pipelines.stage_02 import DataValidationPipeline
from WattPredictor.pipelines.stage_03 import DataTransformationPipeline
from WattPredictor.pipelines.stage_04 import ModelTrainingPipeline
from WattPredictor.pipelines.stage_05 import ModelEvaluationPipeline
from WattPredictor.utils.exception import CustomException



def run_stages(stage_run):
    logger.info(f"Running stage: {stage_run}")
    try:
        if stage_run == "data_ingestion":
            stage = DataIngestionPipeline()
            stage.run()

        elif stage_run == "data_validation":
            stage = DataValidationPipeline()
            stage.run()

        elif stage_run == "data_transformation":
            stage = DataTransformationPipeline()
            stage.run()

        elif stage_run == "model_training":
            stage = ModelTrainingPipeline()
            stage.run()
        
        elif stage_run == "model_evaluation":
            stage = ModelEvaluationPipeline()
            stage.run()

        else:
            logger.info("No valid stage specified. Exiting.")

        logger.info(f"Stage {stage_run} completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e
    

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run Specific Stage of the Pipeline")

        parser.add_argument("--stage")

        args = parser.parse_args()


        if args.stage:
            run_stages(args.stage)

        else:

            stages = [
                "data_ingestion",
                "data_validation",
                "data_transformation",
                "model_training",
                "model_evaluation"
            ]

            for stage in stages:
                run_stages(stage)