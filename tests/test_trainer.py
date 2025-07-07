import pytest

def test_model_training():
    from WattPredictor.config.config import ConfigurationManager
    from WattPredictor.components.model_training import ModelTrainer
    
    config = ConfigurationManager().get_model_trainer_config()
    trainer = ModelTrainer(config)
    result = trainer.train()
    assert "model_name" in result