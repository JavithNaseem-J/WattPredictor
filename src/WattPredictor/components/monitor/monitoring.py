import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
from WattPredictor.utils.logging import logger
from WattPredictor.utils.exception import CustomException
from WattPredictor.entity.config_entity import MonitoringConfig
from WattPredictor.config.inference_config import InferenceConfigurationManager
from WattPredictor.utils.feature import feature_store_instance


class Monitoring:
    def __init__(self,config: MonitoringConfig):
        self.config = config
        self.feature_store = feature_store_instance()

    def load_predictions_and_actuals(self, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        try:
            from_date = from_date.astimezone(timezone.utc)
            to_date = to_date.astimezone(timezone.utc)

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

            combined_df = pd.merge(
                predictions_df,
                actuals_df[['sub_region_code', 'date', 'demand']],
                on=['sub_region_code', 'date'],
                suffixes=('', '_actual')
            )

            mask = (combined_df['date'] >= from_date) & (combined_df['date'] <= to_date)
            filtered_df = combined_df.loc[mask].sort_values(by=['date', 'sub_region_code'])

            logger.info(f"Monitoring data prepared with {len(filtered_df)} records between {from_date} and {to_date}.")
            return filtered_df

        except Exception as e:
            logger.error("Failed to load monitoring data.")
            raise CustomException(e, sys)
