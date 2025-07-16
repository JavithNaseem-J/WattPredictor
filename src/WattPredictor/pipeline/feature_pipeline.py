import argparse
from WattPredictor.config.data_config import DataConfigurationManager
from WattPredictor.components.data.ingestion import Ingestion
from WattPredictor.components.data.validation import Validation
from WattPredictor.components.features.engineering import Engineering


class FeaturePipeline:
    def __init__(self):
        self.data_config = DataConfigurationManager()

    def run(self, step="all"):
        if step in ("all", "ingestion"):
            ingestion_config = self.data_config.get_data_ingestion_config()
            ingestion = Ingestion(config=ingestion_config)
            raw_data = ingestion.download()

        if step in ("all", "validation"):
            data_validation_config = self.data_config.get_data_validation_config()
            data_validation = Validation(data_validation_config)
            data_validation.validator()

        if step in ("all", "engineering"):
            engineering_config = self.data_config.get_data_transformation_config()
            transformation = Engineering(config=engineering_config)
            transformed_data = transformation.transform()
            return transformed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "ingestion", "validation", "engineering"],
        help="Specify which step of the pipeline to run"
    )
    args = parser.parse_args()

    pipeline = FeaturePipeline()
    pipeline.run(step=args.step)
