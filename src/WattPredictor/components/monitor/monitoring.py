import pandas as pd
from datetime import datetime, timedelta
import pytz
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.utils.helpers import create_directories, save_json
from WattPredictor.entity.config_entity import MonitoringConfig
from WattPredictor.utils.logging import logger

class Monitoring:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.feature_store = feature_store_instance()

    def predictions_and_actuals(self):
        logger.info("Starting monitoring process for predictions vs. actuals")
        try:
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
        except Exception as e:
            raise CustomException(f"Failed to load feature groups: {str(e)}", None)

        predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.tz_convert('UTC')
        actuals_df['date'] = pd.to_datetime(actuals_df['date']).dt.tz_convert('UTC')
        actuals_df = actuals_df.rename(columns={'subba': 'sub_region_code', 'value': 'demand'})

        # Log DataFrame details for debugging
        logger.info(f"Predictions DataFrame shape: {predictions_df.shape}")
        logger.info(f"Predictions date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        logger.info(f"Predictions unique sub_region_code: {sorted(predictions_df['sub_region_code'].unique())}")
        logger.info(f"Actuals DataFrame shape: {actuals_df.shape}")
        logger.info(f"Actuals date range: {actuals_df['date'].min()} to {actuals_df['date'].max()}")
        logger.info(f"Actuals unique sub_region_code: {sorted(actuals_df['sub_region_code'].unique())}")

        # Check for common sub_region_code values
        common_codes = set(predictions_df['sub_region_code']).intersection(set(actuals_df['sub_region_code']))
        logger.info(f"Common sub_region_code values: {sorted(common_codes)}")
        if not common_codes:
            logger.warning("No common sub_region_code values between predictions and actuals")

        # Fix: Compare predictions made yesterday with actuals available today
        # Predictions for day D are made on day D-1, actuals for day D are available on day D
        to_date = datetime.now(tz=pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = to_date - timedelta(days=1)

        combined_df = pd.merge(
            predictions_df,
            actuals_df[['sub_region_code', 'date', 'demand']],
            on=['sub_region_code', 'date'],
            how='inner'
        )
        logger.info(f"Combined DataFrame shape after 24-hour merge: {combined_df.shape}")

        try:
            mask = (combined_df['date'] >= from_date) & (combined_df['date'] <= to_date)
            monitoring_df = combined_df.loc[mask].sort_values(by=['sub_region_code', 'date'])
        except Exception as e:
            logger.error(f"Error filtering DataFrame for 24-hour window: {str(e)}")
            raise CustomException(f"Failed to filter DataFrame: {str(e)}", None)


        create_directories([self.config.monitoring_df.parent])
        monitoring_df.to_csv(self.config.monitoring_df, index=False)
        logger.info(f"Monitoring data and metrics saved for {len(monitoring_df)}")
        return monitoring_df