import os
import sys
import json
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (DatasetDriftMetric,ColumnDriftMetric,ColumnSummaryMetric)
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.entity.config_entity import DriftConfig
from WattPredictor.config.data_config import DataConfigurationManager
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.logging import logger


class Drift:
    def __init__(self,config: DriftConfig):
        
        self.config = config
        self.feature_store = feature_store_instance()


    def _load_data(self, start_date, end_date):
        try:
            df, _ = self.feature_store.get_training_data('elec_wx_features_view')
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = df.drop(columns=["date_str"], errors="ignore")
            return df
        except Exception as e:
            raise CustomException(f"Error loading data from Hopsworks: {e}", sys)

    def Detect(self):
        try:
            baseline_df = self._load_data(self.config.baseline_start, self.config.baseline_end)
            current_df = self._load_data(self.config.current_start, self.config.current_end)

            report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
                ColumnDriftMetric(column_name="temperature_2m"),
                ColumnDriftMetric(column_name="sub_region_code"),
                ColumnSummaryMetric(column_name="demand")
            ])

            report.run(reference_data=baseline_df, current_data=current_df)
            create_directories([self.config.report_dir])
            html_path = self.config.report_dir / "drift_report.html"
            json_path = self.config.report_dir / "drift_report.json"

            report.save_html(str(html_path))
            report_dict = report.as_dict()

            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)

            with open(json_path, "w") as f:
                json.dump(report_dict, f, indent=4, default=json_serializer)

            drift_detected = report_dict['metrics'][0]['result'].get('dataset_drift', False)

            logger.info(f"Drift Detected: {drift_detected}")
            logger.info(f"Report saved at {html_path}")
            return drift_detected, report_dict

        except Exception as e:
            raise CustomException(f"Drift detection failed: {e}", sys)