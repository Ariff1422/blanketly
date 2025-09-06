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

def get_seasonal_component(residuals: pd.Series, max_freq_count: int = 1) -> pd.Series:
    """
    Extracts the dominant seasonal component from a series using Fast Fourier Transform (FFT).
    
    Args:
        residuals: A pandas Series containing the residual data.
        max_freq_count: The number of dominant frequencies to use for reconstruction.
    
    Returns:
        A pandas Series of the reconstructed sinusoidal component.
    """
    # Use real FFT since the input is real.
    fft_result = np.fft.rfft(residuals.values)
    
    # Get the frequencies associated with the FFT results.
    frequencies = np.fft.rfftfreq(len(residuals))
    
    # Sort frequencies by the magnitude of their amplitude.
    sorted_freqs = np.argsort(np.abs(fft_result))[-max_freq_count:]
    
    # Create a new array to hold the reconstructed FFT result, with only the dominant frequencies.
    reconstructed_fft = np.zeros_like(fft_result)
    reconstructed_fft[sorted_freqs] = fft_result[sorted_freqs]
    
    # Inverse FFT to get the seasonal component back in the time domain.
    seasonal_component = np.fft.irfft(reconstructed_fft, n=len(residuals))
    
    return pd.Series(seasonal_component, index=residuals.index)

def multi_model_imputation(data: List[Optional[float]]) -> List[float]:
    """
    Imputes missing values in a single time series using a robust multi-model approach.
    
    This function combines a cubic spline trend model with a seasonal component from FFT,
    and a custom autoregressive model for residuals.
    """
    series = pd.Series(data, dtype=float)
    
    known_indices = series.dropna().index
    known_values = series.dropna().values
    
    # Fallback to linear interpolation if not enough data for a spline fit.
    if len(known_indices) < 2:
        return series.interpolate(method='linear', limit_direction='both').fillna(0.0).tolist()
    
    # Step 1: Impute a smooth trend using Cubic Spline Interpolation.
    cs = CubicSpline(known_indices, known_values, bc_type='natural')
    trend_imputed = cs(range(len(series)))

    # Step 2: Calculate residuals after trend removal.
    residuals = known_values - trend_imputed[known_indices]
    
    # Step 3: Extract a seasonal component from the residuals using FFT.
    # We apply FFT to the residuals to find a repeating pattern.
    seasonal_series = pd.Series(np.nan, index=range(len(series)))
    seasonal_series.loc[known_indices] = get_seasonal_component(pd.Series(residuals), max_freq_count=2)
    seasonal_imputed = seasonal_series.interpolate(method='linear', limit_direction='both').values

    # Step 4: Calculate final residuals after both trend and seasonality are removed.
    final_residuals = known_values - trend_imputed[known_indices] - seasonal_imputed[known_indices]
    residual_series = pd.Series(np.nan, index=range(len(series)))
    residual_series[known_indices] = final_residuals

    # Step 5: Custom autoregressive imputation for the final residuals.
    missing_indices = series.isnull().index[series.isnull()]
    
    for i in missing_indices:
        look_back_window_start = max(0, i - 10)
        window_residuals_known = residual_series.iloc[look_back_window_start:i].dropna()
        
        if len(window_residuals_known) > 2:
            x = window_residuals_known.index.values
            y = window_residuals_known.values
            try:
                # Use a quadratic polynomial to model the short-term AR dynamics.
                coeffs = np.polyfit(x, y, 2)
                poly_func = np.poly1d(coeffs)
                predicted_residual = poly_func(i)
                residual_series.loc[i] = predicted_residual
            except np.linalg.LinAlgError:
                residual_series.loc[i] = window_residuals_known.mean()
        else:
            residual_series.loc[i] = window_residuals_known.mean() if len(window_residuals_known) > 0 else 0.0

    # Step 6: Combine all three components: trend, seasonal, and final residuals.
    final_imputed_values = trend_imputed + seasonal_imputed + residual_series.values
    
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
