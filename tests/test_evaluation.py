import pytest

def test_model_evaluation():
    from WattPredictor.config.config import ConfigurationManager
    from WattPredictor.components.model_evaluation import ModelEvaluation

    config = ConfigurationManager().get_model_evaluation_config()
    evaluator = ModelEvaluation(config)
    metrics = evaluator.evaluate()
    assert "mae" in metrics