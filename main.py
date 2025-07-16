import os
import sys
import argparse
from WattPredictor.utils.logging import logger
from WattPredictor.pipeline.feature_pipeline import FeaturePipeline
from WattPredictor.pipeline.training_pipeline import TrainingPipeline
from WattPredictor.pipeline.monitoring_pipeline import MonitoringPipeline
from WattPredictor.utils.exception import CustomException



def run_pipeline(stage: str):
    logger.info(f"Running stage: {stage}")
    try:
        if stage == "feature_pipeline":
            pipeline = FeaturePipeline()
            output = pipeline.run()
            logger.info("Feature Pipeline completed.")

        elif stage == "training_pipeline":
            pipeline = TrainingPipeline()
            output = pipeline.run()
            logger.info("Training Pipeline completed.")

        #elif stage == "inference_pipeline":
            #pipeline = InferencePipeline()
            #output = pipeline.run()
            #logger.info(f"Inference completed. Predictions: {output}")

        elif stage == "monitoring_pipeline":
            pipeline = MonitoringPipeline()
            output = pipeline.run()
            logger.info(f"Monitoring results: {output}")

        else:
            logger.error("Invalid stage specified.")
            output = None

        logger.info(f"Stage {stage} executed successfully")
        return output

    except Exception as e:
        logger.error(f"Pipeline stage {stage} failed: {str(e)}")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipelines for WattPredictor")

    parser.add_argument("--stage", type=str, help="Specify pipeline stage: feature_pipeline, training_pipeline, inference_pipeline, monitoring_pipeline")

    args = parser.parse_args()

    if args.stage:
        run_pipeline(args.stage)

    else:
        stages = [
            "feature_pipeline",
            "training_pipeline",
            "monitoring_pipeline"
        ]

        for stage in stages:
            run_pipeline(stage)
