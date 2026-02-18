from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Dict, Any, List


@dataclass
class WattPredictorConfig:
    
    # ═══════════════════════════════════════════════════════════════════
    # DIRECTORY STRUCTURE
    # ═══════════════════════════════════════════════════════════════════
    
    # config.py is at src/WattPredictor/config/config.py — 4 levels up to project root
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    # ═══════════════════════════════════════════════════════════════════
    # DATA PATHS
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def raw_elec_data_dir(self) -> Path:
        return self.data_dir / "raw" / "elec_data"
    
    @property
    def raw_weather_data_dir(self) -> Path:
        return self.data_dir / "raw" / "wx_data"
    
    @property
    def raw_wx_data_dir(self) -> Path:
        return self.raw_weather_data_dir
    
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def processed_data_path(self) -> Path:
        return self.data_dir / "processed" / "elec_wx_demand.csv"
    
    @property
    def preprocessed_data_path(self) -> Path:
        return self.artifacts_dir / "engineering" / "preprocessed.csv"
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL ARTIFACTS
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "trainer" / "model.joblib"
    
    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "evaluation" / "metrics.json"
    
    @property
    def business_metrics_path(self) -> Path:
        return self.artifacts_dir / "evaluation" / "business_impact.json"
    
    @property
    def predictions_path(self) -> Path:
        return self.artifacts_dir / "prediction" / "predictions.csv"
    
    # ═══════════════════════════════════════════════════════════════════
    # EVIDENTLY MONITORING
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def drift_report_html(self) -> Path:
        return self.artifacts_dir / "drift" / "drift_report.html"
    
    @property
    def drift_report_json(self) -> Path:
        return self.artifacts_dir / "drift" / "drift_report.json"
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════
    
    # Time-series feature configuration
    input_seq_len: int = 672  # 28 days * 24 hours (captures weekly seasonality)
    step_size: int = 1  # Generate features for every hour
    
    # Cross-validation
    cv_folds: int = 3  # Time-series splits for validation
    
    # Model selection grid search parameters
    model_params: Dict[str, Dict[str, List[Any]]] = field(default_factory=lambda: {
        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 7],
            "model__learning_rate": [0.05, 0.1],
        },
        "LightGBM": {
            "model__num_leaves": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [100, 200],
        }
    })
    
    # ═══════════════════════════════════════════════════════════════════
    # API CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    
    # EIA (Energy Information Administration) API
    elec_api: str = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
    elec_api_key: str = field(default_factory=lambda: os.getenv("ELEC_API_KEY", ""))
    
    # Open-Meteo Weather API
    wx_api: str = "https://api.open-meteo.com/v1/forecast"
    
    # NYISO zones configuration
    nyiso_zones: int = 11  # Zones 0-10 (West, Genesee, ..., Long Island)
    
    # ═══════════════════════════════════════════════════════════════════
    # BUSINESS METRICS CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    
    avg_demand_mw: float = 2500  # Average NYISO zone demand
    electricity_price_per_mwh: float = 65  # $/MWh (NYISO average)
    reserve_margin_percent: float = 15  # Industry standard reserve capacity
    peak_capacity_cost_per_mw_year: float = 120000  # Capacity market cost
    
    # ═══════════════════════════════════════════════════════════════════
    # EVIDENTLY MONITORING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    
    # Data drift thresholds
    drift_detection_threshold: float = 0.7  # PSI threshold for drift detection
    
    # Reference window for drift detection
    drift_reference_days: int = 30  # Use last 30 days as reference
    
    # ═══════════════════════════════════════════════════════════════════
    # METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def __post_init__(self):
        directories = [
            self.data_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.raw_elec_data_dir,
            self.raw_weather_data_dir,
            self.processed_data_path.parent,
            self.preprocessed_data_path.parent,
            self.model_path.parent,
            self.metrics_path.parent,
            self.predictions_path.parent,
            self.drift_report_html.parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        # Check hyperparameters first (more specific errors for tests)
        if self.input_seq_len < 24:
            raise ValueError(
                f"input_seq_len must be >= 24, got {self.input_seq_len}"
            )
        
        if self.cv_folds < 2:
            raise ValueError(
                f"cv_folds must be >= 2, got {self.cv_folds}"
            )
        
        # Check API key (can be empty for testing)
        if not self.elec_api_key:
            raise ValueError(
                "ELEC_API_KEY not set. Export it: export ELEC_API_KEY='your_key'"
            )
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_seq_len": self.input_seq_len,
            "step_size": self.step_size,
            "cv_folds": self.cv_folds,
            "nyiso_zones": self.nyiso_zones,
            "model_params": self.model_params,
            "elec_api": self.elec_api,
            "wx_api": self.wx_api,
        }


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════

_config_instance = None


def get_config() -> WattPredictorConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = WattPredictorConfig()
        _config_instance.__post_init__()
    return _config_instance


def reset_config():
    global _config_instance
    _config_instance = None
