import json
from pathlib import Path
from typing import Dict, Any


class BusinessMetrics:
    """Calculate business value and ROI from forecast improvements."""
    
    def __init__(
        self,
        avg_demand_mw: float = 2500,  # Average NYISO zone demand
        electricity_price_per_mwh: float = 65,  # $/MWh (NYISO average)
        reserve_margin_percent: float = 15,  # Industry standard
        peak_capacity_cost_per_mw_year: float = 120000,  # Capacity market cost
    ):
        self.avg_demand_mw = avg_demand_mw
        self.electricity_price = electricity_price_per_mwh
        self.reserve_margin = reserve_margin_percent / 100
        self.capacity_cost = peak_capacity_cost_per_mw_year
    
    def calculate_baseline_costs(self) -> Dict[str, float]:
        """Calculate current operational costs without ML forecasting."""
        # Conservative estimate: 10% forecasting error (industry baseline)
        baseline_error_mw = self.avg_demand_mw * 0.10
        
        # Reserve capacity costs (over-provisioning due to uncertainty)
        reserve_capacity_mw = self.avg_demand_mw * self.reserve_margin
        annual_reserve_cost = reserve_capacity_mw * self.capacity_cost
        
        # Energy imbalance costs (buying energy last-minute at premium)
        hours_per_year = 8760
        imbalance_penalty_multiplier = 1.5 
        annual_imbalance_cost = (
            baseline_error_mw * 
            self.electricity_price * 
            imbalance_penalty_multiplier * 
            hours_per_year
        )
        
        total_baseline_cost = annual_reserve_cost + annual_imbalance_cost
        
        return {
            "baseline_forecast_error_mw": baseline_error_mw,
            "baseline_reserve_capacity_mw": reserve_capacity_mw,
            "annual_reserve_cost_usd": annual_reserve_cost,
            "annual_imbalance_cost_usd": annual_imbalance_cost,
            "total_annual_cost_usd": total_baseline_cost
        }
    
    def calculate_ml_improvements(
        self, 
        rmse: float,
        mae: float,
        mape: float
    ) -> Dict[str, Any]:
        """Calculate cost savings with ML-based forecasting."""
        baseline = self.calculate_baseline_costs()
        
        # ML model reduces error from 10% (baseline) to MAPE%
        baseline_error_percent = 10.0
        ml_error_percent = mape
        error_reduction_percent = ((baseline_error_percent - ml_error_percent) / baseline_error_percent) * 100
        
        # New error in MW
        ml_forecast_error_mw = self.avg_demand_mw * (ml_error_percent / 100)
        
        # Reduced reserve margin needed (better forecasts = less buffer)
        reduced_reserve_margin = self.reserve_margin * (ml_error_percent / baseline_error_percent)
        new_reserve_capacity_mw = self.avg_demand_mw * reduced_reserve_margin
        reserve_savings_mw = baseline["baseline_reserve_capacity_mw"] - new_reserve_capacity_mw
        annual_reserve_savings = reserve_savings_mw * self.capacity_cost
        
        # Reduced imbalance costs
        hours_per_year = 8760
        imbalance_penalty_multiplier = 1.5
        new_imbalance_cost = (
            ml_forecast_error_mw * 
            self.electricity_price * 
            imbalance_penalty_multiplier * 
            hours_per_year
        )
        imbalance_savings = baseline["annual_imbalance_cost_usd"] - new_imbalance_cost
        
        total_annual_savings = annual_reserve_savings + imbalance_savings
        
        # Per-hour savings (to show how hourly predictions add up)
        hours_per_year = 8760
        per_hour_savings = total_annual_savings / hours_per_year
        
        # ROI calculation (assuming $200k investment in ML infrastructure)
        ml_infrastructure_cost = 200000
        roi_years = ml_infrastructure_cost / total_annual_savings if total_annual_savings > 0 else float('inf')
        roi_percent = (total_annual_savings / ml_infrastructure_cost) * 100 if ml_infrastructure_cost > 0 else 0
        
        return {
            "model_metrics": {
                "rmse_mw": rmse,
                "mae_mw": mae,
                "mape_percent": mape
            },
            "forecast_improvement": {
                "baseline_error_percent": baseline_error_percent,
                "ml_error_percent": ml_error_percent,
                "error_reduction_percent": error_reduction_percent,
                "ml_forecast_error_mw": ml_forecast_error_mw
            },
            "cost_savings": {
                "reserve_capacity_savings_mw": reserve_savings_mw,
                "annual_reserve_savings_usd": annual_reserve_savings,
                "annual_imbalance_savings_usd": imbalance_savings,
                "total_annual_savings_usd": total_annual_savings,
                "per_hour_savings_usd": per_hour_savings,
                "hours_per_year": hours_per_year
            },
            "roi": {
                "ml_infrastructure_investment_usd": ml_infrastructure_cost,
                "roi_payback_years": roi_years,
                "roi_percent_annual": roi_percent
            },
            "baseline_comparison": baseline
        }
    
    def generate_report(
        self,
        rmse: float,
        mae: float,
        mape: float,
        output_path: Path = None
    ) -> Dict[str, Any]:
        """Generate comprehensive business impact report."""
        results = self.calculate_ml_improvements(rmse, mae, mape)
        
        # Format for human readability
        report = {
            "executive_summary": {
                "per_hour_savings_usd": f"${results['cost_savings']['per_hour_savings_usd']:.2f}/hour",
                "annual_cost_savings_usd": f"${results['cost_savings']['total_annual_savings_usd']:,.0f}/year",
                "roi_payback_period": f"{results['roi']['roi_payback_years']:.1f} years",
                "forecast_accuracy_improvement": f"{results['forecast_improvement']['error_reduction_percent']:.1f}%",
                "grid_capacity_freed_mw": f"{results['cost_savings']['reserve_capacity_savings_mw']:.0f} MW"
            },
            "detailed_results": results
        }
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report


def demo_calculation():
    """Demo calculation with example metrics."""
    calculator = BusinessMetrics(
        avg_demand_mw=2500,  # NYISO zone average
        electricity_price_per_mwh=65,
        reserve_margin_percent=15,
        peak_capacity_cost_per_mw_year=120000
    )
    
    # Example: Model with 3.5% MAPE
    report = calculator.generate_report(
        rmse=85.0,
        mae=55.0,
        mape=3.5,
        output_path=Path("artifacts/business_impact_report.json")
    )
    
    print("=" * 60)
    print("WATTPREDICTOR - BUSINESS IMPACT ANALYSIS")
    print("=" * 60)
    print(f"💰 Annual Cost Savings: {report['executive_summary']['annual_cost_savings_usd']}")
    print(f"📊 ROI Payback Period: {report['executive_summary']['roi_payback_period']}")
    print(f"📈 Forecast Improvement: {report['executive_summary']['forecast_accuracy_improvement']}")
    print(f"⚡ Grid Capacity Freed: {report['executive_summary']['grid_capacity_freed_mw']}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    demo_calculation()
