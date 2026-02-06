# Config package - unified configuration management
from WattPredictor.config.config_manager import ConfigManager

# Legacy aliases for backward compatibility
from WattPredictor.config.config_manager import (
    DataConfigurationManager,
    ModelConfigurationManager,
    InferenceConfigurationManager,
    FeatureConfigurationManager
)

__all__ = [
    'ConfigManager',
    'DataConfigurationManager',
    'ModelConfigurationManager', 
    'InferenceConfigurationManager',
    'FeatureConfigurationManager'
]
