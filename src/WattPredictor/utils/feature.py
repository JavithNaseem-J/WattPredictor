from functools import lru_cache
from WattPredictor.config.config_manager import ConfigManager
from WattPredictor.components.features.feature_store import FeatureStore


@lru_cache(maxsize=1)
def feature_store_instance() -> FeatureStore:
    """Get or create singleton FeatureStore instance."""
    config = ConfigManager()
    feature_store_config = config.get_feature_store_config()
    return FeatureStore(config=feature_store_config)