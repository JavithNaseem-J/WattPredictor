import pandas as pd
from datetime import datetime, timedelta
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.config.inference_config import InferenceConfigurationManager
from WattPredictor.utils.helpers import create_directories
from WattPredictor.entity.config_entity import MonitoringConfig
from WattPredictor.utils.logging import logger

class Monitoring:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.feature_store = feature_store_instance()

    def predictions_and_actuals(self):
        predictions_fg = self.feature_store.feature_store.get_feature_group(
            name=self.config.predictions_fg_name,
            version=self.config.predictions_fg_version
        )
        actuals_fg = self.feature_store.feature_store.get_feature_group(
            name=self.config.actuals_fg_name,
            version=self.config.actuals_fg_version
        )
        predictions_df = predictions_fg.read()
        actuals_df = actuals_fg.read()

        predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.tz_convert('UTC')
        actuals_df['date'] = pd.to_datetime(actuals_df['date']).dt.tz_convert('UTC')

        from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
        to_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d 23:59:59")

        combined_df = pd.merge(
            predictions_df,
            actuals_df[['sub_region_code', 'date', 'demand']],
            on=['sub_region_code', 'date']
        )
        mask = (combined_df['date'] >= from_date) & (combined_df['date'] <= to_date)
        monitoring_df = combined_df.loc[mask].sort_values(by=['date', 'sub_region_code'])

        create_directories([self.config.monitoring_df.parent])
        monitoring_df.to_csv(self.config.monitoring_df, index=False)
        logger.info(f"Monitoring data saved to {self.config.monitoring_df}")
        return monitoring_df