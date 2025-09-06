import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from scipy.interpolate import CubicSpline
from statsmodels.tsa.ar_model import AutoReg

# The main FastAPI application instance.
app = FastAPI()

# Pydantic model for the request body.
class SeriesData(BaseModel):
    series: List[List[Optional[float]]] = Field(..., description="An array of 100 lists, each with 1000 elements (float or null).")

def multi_model_imputation(data: List[Optional[float]]) -> List[float]:
    """
    Imputes missing values in a single time series using a robust multi-model approach.
    
    This function combines a cubic spline trend model with a dynamically-lagged
    autoregressive (AR) model to handle both long-term trends and short-term dependencies.
    """
    # Convert list to a pandas Series for easy handling of NaNs.
    series = pd.Series(data, dtype=float)
    
    # Get indices and values of non-null data points.
    known_indices = series.dropna().index
    known_values = series.dropna().values
    
    # If there is not enough data to model, return a fallback using linear interpolation.
    if len(known_indices) < 2:
        return series.interpolate(method='linear', limit_direction='both').fillna(0.0).tolist()
    
    # Step 1: Impute a smooth trend using Cubic Spline Interpolation.
    # This captures the polynomial, sinusoidal, or overall trend component.
    cs = CubicSpline(known_indices, known_values, bc_type='natural')
    trend_imputed = cs(range(len(series)))

    # Step 2: Calculate residuals on known data points.
    # These residuals represent the noise and any short-term autoregressive behavior.
    residuals = known_values - trend_imputed[known_indices]

    # Step 3: Find the optimal lag for the AR model using the Bayesian Information Criterion (BIC).
    # This makes the model adaptive to different time series. We limit the max lag to 10
    # to prevent overfitting and ensure performance.
    best_bic = np.inf
    best_lag = 1
    
    # The number of data points for fitting the AR model
    nobs = len(residuals)
    max_lag = min(10, nobs - 1)

    for lag in range(1, max_lag + 1):
        try:
            model = AutoReg(residuals, lags=lag, old_names=False)
            model_fit = model.fit()
            bic = model_fit.bic
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except (ValueError, np.linalg.LinAlgError):
            continue

    # Step 4: Fit the final AR model using the best lag found.
    # If no suitable lag was found, fall back gracefully.
    final_imputed_values = trend_imputed.copy()
    try:
        model = AutoReg(residuals, lags=best_lag, old_names=False)
        model_fit = model.fit()
        
        # Step 5: Use the AR model to forecast residuals for the missing points.
        missing_indices = series.isnull().index[series.isnull()]
        forecasted_residuals = model_fit.forecast(steps=len(missing_indices))
        
        # Step 6: Combine the trend and the forecasted residuals for final imputation.
        if len(forecasted_residuals) == len(missing_indices):
            final_imputed_values[missing_indices] = trend_imputed[missing_indices] + forecasted_residuals.values
        
    except (ValueError, np.linalg.LinAlgError, IndexError, Exception):
        # Fallback to linear interpolation if the AR model fails to fit or predict.
        print("An error occurred during AR model fitting/prediction. Falling back to linear interpolation.")
        final_imputed_values = series.interpolate(method='linear', limit_direction='both').values
    
    # Clamp values to avoid instability or NaNs/Infs that could result from complex models.
    final_imputed_values = np.nan_to_num(final_imputed_values, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return final_imputed_values.tolist()

@app.post("/blankety")
async def blankety(data: SeriesData):
    """
    POST endpoint to process the time series data and impute missing values.
    """
    completed_series = []
    # Process each time series in the input.
    for series_list in data.series:
        imputed_list = multi_model_imputation(series_list)
        completed_series.append(imputed_list)
    
    return {"answer": completed_series}

if __name__ == "__main__":
    # Start the Uvicorn server.
    uvicorn.run(app, host="0.0.0.0", port=8000)
