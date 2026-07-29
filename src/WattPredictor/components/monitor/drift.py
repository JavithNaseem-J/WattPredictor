import os
import json
import pandas as pd
from datetime import timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (DatasetDriftMetric, ColumnDriftMetric, ColumnSummaryMetric)
from WattPredictor.config.config import WattPredictorConfig, get_config
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.logging import logger


class Drift:
    def __init__(self, config: WattPredictorConfig = None):
        self.config = config or get_config()

    def _load_data(self, start_date, end_date):
        preprocessed_path = str(self.config.preprocessed_data_path)
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")
        df = pd.read_csv(preprocessed_path)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        df = df.drop(columns=["date_str"], errors="ignore")
        return df

    def Detect(self):
        try:
            preprocessed_path = str(self.config.preprocessed_data_path)
            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")
            
            raw_df = pd.read_csv(preprocessed_path)
            max_date = pd.to_datetime(raw_df['date'], utc=True).dt.tz_localize(None).max()

            baseline_start = max_date - timedelta(days=365)
            baseline_end = max_date - timedelta(days=30)
            current_start = max_date - timedelta(days=29)
            current_end = max_date

            baseline_df = self._load_data(baseline_start, baseline_end)
            current_df = self._load_data(current_start, current_end)

            report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
                ColumnDriftMetric(column_name="temperature_2m"),
                ColumnDriftMetric(column_name="sub_region_code"),
                ColumnSummaryMetric(column_name="demand")
            ])

            report.run(reference_data=baseline_df, current_data=current_df)
            create_directories([self.config.drift_report_dir])
            html_path = self.config.drift_report_html
            json_path = self.config.drift_report_json

            report.save_html(str(html_path))
            report_dict = report.as_dict()

            with open(json_path, "w") as f:
                json.dump(report_dict, f, indent=4, default=str)

            drift_detected = report_dict['metrics'][0]['result'].get('dataset_drift', False)

            logger.info(f"Drift Detected: {drift_detected}")
            logger.info(f"Report saved at {html_path}")
            return drift_detected, report_dict

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise