import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from WattPredictor.config.config import get_config
from WattPredictor.components.features.ingestion import Ingestion
from WattPredictor.components.features.validation import Validation
from WattPredictor.components.features.engineering import Engineering
from WattPredictor.entity.config_entity import IngestionConfig, ValidationConfig, EngineeringConfig
from WattPredictor.utils.logging import logger


class FeaturePipeline:
    
    def __init__(self):
        self.config = get_config()
    
    def run(self):
        try:
            # Step 1: Ingestion
            logger.info("=" * 60)
            logger.info("STEP 1: Data Ingestion")
            logger.info("=" * 60)
            ingestion_config = IngestionConfig(
                root_dir=self.config.data_dir,
                elec_api=self.config.elec_api,
                elec_api_key=self.config.elec_api_key,
                wx_api=self.config.wx_api,
                elec_raw_data=self.config.raw_elec_data_dir,
                wx_raw_data=self.config.raw_wx_data_dir,
                data_file=self.config.processed_data_path
            )
            ingestor = Ingestion(config=ingestion_config)
            ingestor.download()
            logger.info("Data ingestion completed")
            
            # Step 2: Validation
            logger.info("=" * 60)
            logger.info("STEP 2: Data Validation")
            logger.info("=" * 60)
            validation_config = ValidationConfig(
                root_dir=self.config.artifacts_dir / "validation",
                data_file=ingestion_config.data_file,
                status_file=self.config.artifacts_dir / "validation" / "status.json",
                all_schema={}  # Schema validation not used in simplified version
            )
            validator = Validation(config=validation_config)
            validator.validator()
            logger.info("Data validation completed")
            
            # Step 3: Feature Engineering
            logger.info("=" * 60)
            logger.info("STEP 3: Feature Engineering")
            logger.info("=" * 60)
            engineering_config = EngineeringConfig(
                root_dir=self.config.artifacts_dir / "engineering",
                data_file=validation_config.data_file,
                status_file=self.config.artifacts_dir / "validation" / "status.json",
                preprocessed=self.config.preprocessed_data_path
            )
            transformer = Engineering(config=engineering_config)
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
