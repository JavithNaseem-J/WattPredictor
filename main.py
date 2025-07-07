import os
import sys
import argparse
from WattPredictor import logger
from WattPredictor.utils.exception import CustomException
from WattPredictor.pipelines.stage_01 import DataIngestionPipeline
from WattPredictor.pipelines.stage_02 import DataValidationPipeline
from WattPredictor.pipelines.stage_03 import DataTransformationPipeline
from WattPredictor.pipelines.stage_04 import ModelTrainingPipeline
from WattPredictor.pipelines.stage_05 import ModelEvaluationPipeline


def run_stages(stage_run):
    try:
        if stage_run == "data_ingestion":
            logger.info("Starting Data Ingestion Stage")
            satge = DataIngestionPipeline()
            satge.run()
            logger.info("Data Ingestion Stage completed successfully")

        elif stage_run == "data_validation":
            logger.info("Starting Data Validation Stage")
            satge = DataValidationPipeline()
            satge.run()
            logger.info("Data Validation Stage completed successfully")

        elif stage_run == "data_transformation":
            logger.info("Starting Data Transformation Stage")
            satge = DataTransformationPipeline()
            satge.run()
            logger.info("Data Transformation Stage completed successfully")

        elif stage_run == "model_training":
            logger.info("Starting Model Training Stage")
            satge = ModelTrainingPipeline()
            satge.run()
            logger.info("Model Training Stage completed successfully")
        
        elif stage_run == "model_evaluation":
            logger.info("Starting Model Evaluation Stage")
            satge = ModelEvaluationPipeline()
            satge.run()
            logger.info("Model Evaluation Stage completed successfully")

        else:
            logger.info("No valid stage specified. Exiting.")

    except Exception as e:
        raise CustomException(e, sys) from e
    

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Ru Specific Stage of the Pipeline")

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