import pytest

def test_data_transformation():
    from WattPredictor.config.config import ConfigurationManager
    from WattPredictor.components.data_transformation import DataTransformation
    
    config = ConfigurationManager().get_data_transformation_config()
    transformer = DataTransformation(config)
    train_df, test_df = transformer.train_test_splitting()
    assert not train_df.empty and not test_df.empty
    (train_x, train_y), (test_x, test_y) = transformer.preprocess_features(train_df, test_df)
    assert not train_x.empty and not train_y.empty
    assert not test_x.empty and not test_y.empty  