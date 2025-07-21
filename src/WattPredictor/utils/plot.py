import pandas as pd
import plotly.graph_objects as go

def plot_one_sample(example_id: int, features: pd.DataFrame, targets: pd.Series, predictions: pd.Series, display_title: bool = True) -> go.Figure:
    """
    Args:
        example_id (int): Index of the sample to plot (row index in features/targets).
        features (pd.DataFrame): DataFrame containing demand_previous_{1 to 672}_hour columns.
        targets (pd.Series): Series of target values (predicted_demand).
        predictions (pd.Series): Series of predicted values (predicted_demand).
        display_title (bool): Whether to display the plot title.

    Returns:
        go.Figure: Plotly figure object with the time-series plot.
    """
    demand_cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(672))]
    demand_series = features.iloc[example_id][demand_cols].values

    hours = pd.date_range(end=features["date"].iloc[example_id], periods=672, freq="H", tz="UTC")
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=demand_series,
            mode="lines",
            name="Historical Demand",
            line=dict(color="blue")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[features["date"].iloc[example_id]],
            y=[predictions.iloc[example_id]],
            mode="markers",
            name="Predicted Demand",
            marker=dict(color="red", size=10)
        )
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand (MW)",
        showlegend=True,
        template="plotly_white"
    )
    
    if display_title:
        fig.update_layout(title=f"Demand for sub_region_code {features['sub_region_code'].iloc[example_id]}")
    
    return fig