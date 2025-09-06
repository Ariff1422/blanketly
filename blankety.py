import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from scipy.interpolate import CubicSpline

# The main FastAPI application instance.
app = FastAPI()

# Pydantic model for the request body.
class SeriesData(BaseModel):
    series: List[List[Optional[float]]] = Field(..., description="An array of 100 lists, each with 1000 elements (float or null).")

def multi_model_imputation(data: List[Optional[float]]) -> List[float]:
    """
    Imputes missing values in a single time series using a robust multi-model approach.
    
    This function combines a cubic spline trend model with a custom autoregressive (AR)
    model to handle both long-term trends and short-term dependencies, without using statsmodels.
    """
    series = pd.Series(data, dtype=float)
    
    known_indices = series.dropna().index
    known_values = series.dropna().values
    
    # Fallback to linear interpolation if not enough data is available for a spline fit.
    if len(known_indices) < 2:
        return series.interpolate(method='linear', limit_direction='both').fillna(0.0).tolist()
    
    # Step 1: Impute a smooth trend using Cubic Spline Interpolation.
    cs = CubicSpline(known_indices, known_values, bc_type='natural')
    trend_imputed = cs(range(len(series)))

    # Step 2: Calculate residuals on known data points.
    residuals = known_values - trend_imputed[known_indices]
    residual_series = pd.Series(np.nan, index=range(len(series)))
    residual_series[known_indices] = residuals

    # Step 3: Custom autoregressive imputation for residuals.
    # This replaces the functionality of statsmodels.
    missing_indices = series.isnull().index[series.isnull()]
    
    for i in missing_indices:
        # Define a look-back window to model short-term dependencies.
        # We'll use a window of up to 10 points.
        look_back_window_start = max(0, i - 10)
        
        # Get the non-null residuals in the look-back window.
        window_residuals_known = residual_series.iloc[look_back_window_start:i].dropna()
        
        if len(window_residuals_known) > 1:
            # Fit a simple linear model to the known residuals in the window.
            x = window_residuals_known.index.values
            y = window_residuals_known.values
            
            try:
                # Use numpy.polyfit for a simple linear regression (polynomial of degree 1).
                # This models the AR process without external libraries.
                coeffs = np.polyfit(x, y, 1)
                poly_func = np.poly1d(coeffs)
                predicted_residual = poly_func(i)
                residual_series.loc[i] = predicted_residual
            except np.linalg.LinAlgError:
                # Fallback if polyfit fails.
                residual_series.loc[i] = window_residuals_known.mean()
        else:
            # If there's not enough history, use a simple mean or 0.
            residual_series.loc[i] = window_residuals_known.mean() if len(window_residuals_known) > 0 else 0.0

    # Step 4: Combine the trend and the imputed residuals.
    final_imputed_values = trend_imputed + residual_series.values
    
    # Clamp values to avoid instability or NaNs/Infs that could result from complex calculations.
    final_imputed_values = np.nan_to_num(final_imputed_values, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return final_imputed_values.tolist()

@app.post("/blankety")
async def blankety(data: SeriesData):
    """
    POST endpoint to process the time series data and impute missing values.
    """
    completed_series = []
    for series_list in data.series:
        imputed_list = multi_model_imputation(series_list)
        completed_series.append(imputed_list)
    
    return {"answer": completed_series}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
