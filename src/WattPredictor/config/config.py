from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml
from typing import Dict, Any, List


@dataclass
class WattPredictorConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    _yaml_config: Dict[str, Any] = field(default_factory=dict, init=False)
    _yaml_params: Dict[str, Any] = field(default_factory=dict, init=False)
    
    # ═══════════════════════════════════════════════════════════════════
    # INITIALIZATION & YAML LOADING
    # ═══════════════════════════════════════════════════════════════════
    
    def __post_init__(self):
        config_path = self.project_root / "config_file" / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self._yaml_config = yaml.safe_load(f) or {}
                
        params_path = self.project_root / "config_file" / "params.yaml"
        if params_path.exists():
            with open(params_path, encoding="utf-8") as f:
                self._yaml_params = yaml.safe_load(f) or {}
                
        directories = [
            self.data_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.raw_elec_data_dir,
            self.raw_wx_data_dir,
            self.processed_data_path.parent,
            self.preprocessed_data_path.parent,
            self.model_path.parent,
            self.metrics_path.parent,
            self.predictions_path.parent,
            self.drift_report_dir,
            self.monitoring_df_path.parent,
            self.status_file.parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    # ═══════════════════════════════════════════════════════════════════
    # DIRECTORY STRUCTURE
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def data_dir(self) -> Path:
        rel = self._yaml_config.get("data", {}).get("root_dir", "data")
        return self.project_root / rel
    
    @property
    def artifacts_dir(self) -> Path:
        rel = self._yaml_config.get("artifacts_root", "artifacts")
        return self.project_root / rel
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    # ═══════════════════════════════════════════════════════════════════
    # DATA PATHS
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def raw_elec_data_dir(self) -> Path:
        rel = self._yaml_config.get("data", {}).get("elec_raw_data", "data/raw/elec_data")
        return self.project_root / rel
    
    @property
    def raw_weather_data_dir(self) -> Path:
        rel = self._yaml_config.get("data", {}).get("wx_raw_data", "data/raw/wx_data")
        return self.project_root / rel
    
    @property
    def raw_wx_data_dir(self) -> Path:
        return self.raw_weather_data_dir
    
    @property
    def processed_data_path(self) -> Path:
        rel = self._yaml_config.get("data", {}).get("data_file", "data/processed/elec_wx_demand.csv")
        return self.project_root / rel
    
    @property
    def preprocessed_data_path(self) -> Path:
        rel = self._yaml_config.get("engineering", {}).get("preprocessed", "artifacts/engineering/preprocessed.csv")
        return self.project_root / rel
    
    @property
    def status_file(self) -> Path:
        rel = self._yaml_config.get("validation", {}).get("status_file", "artifacts/validation/status.json")
        return self.project_root / rel
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL ARTIFACTS
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def model_path(self) -> Path:
        trainer_cfg = self._yaml_config.get("trainer", {})
        root = trainer_cfg.get("root_dir", "artifacts/trainer")
        name = trainer_cfg.get("model_name", "model.joblib")
        return self.project_root / root / name
    
    @property
    def metrics_path(self) -> Path:
        rel = self._yaml_config.get("evaluation", {}).get("metrics_path", "artifacts/evaluation/metrics.json")
        return self.project_root / rel
    
    @property
    def img_path(self) -> Path:
        rel = self._yaml_config.get("evaluation", {}).get("img_path", "artifacts/evaluation/pred_vs_actual.png")
        return self.project_root / rel
    
    @property
    def predictions_path(self) -> Path:
        rel = self._yaml_config.get("prediction", {}).get("predictions_df", "artifacts/prediction/predictions.csv")
        return self.project_root / rel
    
    @property
    def monitoring_df_path(self) -> Path:
        rel = self._yaml_config.get("monitoring", {}).get("monitoring_df", "artifacts/monitoring/monitoring_df.csv")
        return self.project_root / rel
    
    # ═══════════════════════════════════════════════════════════════════
    # EVIDENTLY MONITORING
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def drift_report_dir(self) -> Path:
        rel = self._yaml_config.get("drift", {}).get("report_dir", "artifacts/drift")
        return self.project_root / rel
    
    @property
    def drift_report_html(self) -> Path:
        return self.drift_report_dir / "drift_report.html"
    
    @property
    def drift_report_json(self) -> Path:
        return self.drift_report_dir / "drift_report.json"
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def input_seq_len(self) -> int:
        return self._yaml_params.get("training", {}).get("input_seq_len", 672)
    
    @property
    def step_size(self) -> int:
        return self._yaml_params.get("training", {}).get("step_size", 1)
    
    @property
    def cv_folds(self) -> int:
        return self._yaml_params.get("training", {}).get("cv_folds", 3)
    
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
    
    elec_api: str = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
    elec_api_key: str = field(default_factory=lambda: os.getenv("ELEC_API_KEY", ""))
    wx_api: str = "https://api.open-meteo.com/v1/forecast"
    nyiso_zones: int = 11
    
    # ═══════════════════════════════════════════════════════════════════
    # BUSINESS METRICS CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    
    avg_demand_mw: float = 2500
    electricity_price_per_mwh: float = 65
    reserve_margin_percent: float = 15
    peak_capacity_cost_per_mw_year: float = 120000
    
    # ═══════════════════════════════════════════════════════════════════
    # METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def validate(self) -> bool:
        if self.input_seq_len < 24:
            raise ValueError(f"input_seq_len must be >= 24, got {self.input_seq_len}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        if not self.elec_api_key:
            raise ValueError("ELEC_API_KEY not set. Export it: export ELEC_API_KEY='your_key'")
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


_config_instance = None


def get_config() -> WattPredictorConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = WattPredictorConfig()
    return _config_instance


def reset_config():
    global _config_instance
    _config_instance = None
