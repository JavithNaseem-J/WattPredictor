import pytest

def test_data_validation():
    from WattPredictor.config.config import ConfigurationManager
    from WattPredictor.components.data_validation import DataValidation

    config = ConfigurationManager().get_data_validation_config()
    validation = DataValidation(config)
    status = validation.validation()
    assert isinstance(status, bool)