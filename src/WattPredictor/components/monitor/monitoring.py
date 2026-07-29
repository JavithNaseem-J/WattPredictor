import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from WattPredictor.config.config import WattPredictorConfig, get_config
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.logging import logger


class Monitoring:
    def __init__(self, config: WattPredictorConfig = None):
        self.config = config or get_config()

    def predictions_and_actuals(self):
        logger.info("Starting monitoring process for predictions vs. actuals")
        predictions_path = str(self.config.predictions_path)
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(f"Predictions not found: {predictions_path}")
        predictions_df = pd.read_csv(predictions_path)
        
        actuals_path = str(self.config.preprocessed_data_path)
        if not os.path.exists(actuals_path):
            raise FileNotFoundError(f"Actuals not found: {actuals_path}")
        actuals_df = pd.read_csv(actuals_path)

        predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.tz_convert('UTC')
        actuals_df['date'] = pd.to_datetime(actuals_df['date']).dt.tz_convert('UTC')
        actuals_df = actuals_df.rename(columns={'subba': 'sub_region_code', 'value': 'demand'})

        logger.info(f"Predictions DataFrame shape: {predictions_df.shape}")
        logger.info(f"Predictions date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        logger.info(f"Actuals DataFrame shape: {actuals_df.shape}")
        logger.info(f"Actuals date range: {actuals_df['date'].min()} to {actuals_df['date'].max()}")

        common_codes = set(predictions_df['sub_region_code']).intersection(set(actuals_df['sub_region_code']))
        logger.info(f"Common sub_region_code values: {sorted(common_codes)}")

        to_date = datetime.now(tz=pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = to_date - timedelta(days=1)

        combined_df = pd.merge(
            predictions_df,
            actuals_df[['sub_region_code', 'date', 'demand']],
            on=['sub_region_code', 'date'],
            how='inner'
        )
        logger.info(f"Combined DataFrame shape after 24-hour merge: {combined_df.shape}")

        mask = (combined_df['date'] >= from_date) & (combined_df['date'] <= to_date)
        monitoring_df = combined_df.loc[mask].sort_values(by=['sub_region_code', 'date'])

        create_directories([self.config.monitoring_df_path.parent])
        monitoring_df.to_csv(self.config.monitoring_df_path, index=False)
        logger.info(f"Monitoring data and metrics saved for {len(monitoring_df)} records")
        return monitoring_df