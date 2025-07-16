from functools import lru_cache
from WattPredictor.config.feature_config import FeatureConfigurationManager
from WattPredictor.components.features.feature_store import FeatureStore

@lru_cache(maxsize=1)
def feature_store_instance() -> FeatureStore:
    feature_config = FeatureConfigurationManager()
    feature_store_config = feature_config.get_feature_store_config()
    return FeatureStore(config=feature_store_config)