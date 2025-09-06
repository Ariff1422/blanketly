import numpy as np
from scipy.interpolate import CubicSpline
from fastapi import FastAPI, Request, HTTPException
import json

# This class contains the core logic for imputing missing values.
class BlanketyBlanks:
    """
    A class to solve the Blankety Blanks hackathon problem by imputing
    missing values in time series data.
    """

    def _get_max_decimal_places(self, data: list) -> int:
        """
        Helper method to determine the maximum number of decimal places
        in a list of numeric data.
        """
        max_dp = 0
        for item in data:
            if isinstance(item, (float, int)):
                s = str(item)
                if '.' in s:
                    decimal_places = len(s.split('.')[1])
                    if decimal_places > max_dp:
                        max_dp = decimal_places
        return max_dp

    def impute_missing_values(self, input_data: dict) -> dict:
        """
        Processes a dictionary of time series, imputing all null values
        using cubic spline interpolation and matching decimal precision.

        Args:
            input_data (dict): A dictionary with a "series" key containing
                               a list of 100 lists of floats or None.

        Returns:
            dict: A dictionary with an "answer" key containing the 100
                  completed series with no null values.
        """
        try:
            series_list = input_data.get("series", [])
            if not series_list:
                raise ValueError("Input JSON is missing the 'series' key or is empty.")

            completed_series = []

            # Iterate through each of the 100 series
            for series in series_list:
                # Get max decimal places from original data
                max_dp = self._get_max_decimal_places(series)
                
                # Convert list to a NumPy array for efficient processing.
                # Replace None with np.nan for numerical operations.
                series_np = np.array(series, dtype=np.float64)
                series_np[series_np == None] = np.nan

                # Find the indices of the non-null and null values.
                not_null_indices = np.where(~np.isnan(series_np))[0]
                null_indices = np.where(np.isnan(series_np))[0]
                
                # If there are no missing values, just return the original series.
                if len(null_indices) == 0:
                    completed_series.append(series_np.tolist())
                    continue

                # Ensure we have at least 2 non-null points for interpolation.
                if len(not_null_indices) < 2:
                    # If not enough data, fill with 0 to be safe.
                    series_np[null_indices] = 0.0
                    completed_series.append(series_np.tolist())
                    continue

                # Get the x (indices) and y (values) for interpolation.
                x_known = not_null_indices
                y_known = series_np[not_null_indices]

                # Create a cubic spline based on the non-null values.
                cs = CubicSpline(x_known, y_known, bc_type='natural')
                
                # Predict the values for the null indices.
                imputed_values = cs(null_indices)
                
                # Round imputed values to match the precision of the input data.
                imputed_values = np.round(imputed_values, decimals=max_dp)

                # Replace the NaN values with the imputed values.
                series_np[null_indices] = imputed_values

                # Clamp values to prevent infinities or very large numbers
                series_np = np.clip(series_np, -1e10, 1e10)

                # Convert the NumPy array back to a list of floats.
                completed_series.append(series_np.tolist())

            return {"answer": completed_series}

        except Exception as e:
            # General error handling for unexpected issues.
            print(f"An error occurred during imputation: {e}")
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# This part sets up the FastAPI application to handle the requests.
app = FastAPI()
solver = BlanketyBlanks()

@app.post("/blankety")
async def handle_blankety_request(request: Request):
    """
    Exposes a POST endpoint at '/blankety' to solve the hackathon challenge.
    It takes a JSON payload, imputes missing values, and returns the result.
    """
    if not request.headers.get('content-type') == 'application/json':
        raise HTTPException(status_code=400, detail="Request must be JSON")

    try:
        input_data = await request.json()
        if not input_data or "series" not in input_data:
            raise HTTPException(status_code=400, detail="Invalid JSON payload. 'series' key is missing.")

        # Call the imputation logic
        result = solver.impute_missing_values(input_data)
        
        # Return the result as a JSON response
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
