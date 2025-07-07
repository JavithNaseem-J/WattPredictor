import pytest
from pathlib import Path

def test_data_ingestion():
    from WattPredictor.config.config import ConfigurationManager
    from WattPredictor.components.data_ingestion import DataIngestion

    config = ConfigurationManager().get_data_ingestion_config()
    ingestion = DataIngestion(config)
    ingestion.download()
    assert Path(config.data_file).exists()
    print("✅ Data ingestion test passed.")