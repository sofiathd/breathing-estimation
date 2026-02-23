import numpy as np

def fit_linear_calibration(x, y):
    """
    Fit y=ax+b using least squares.
    
    Args:
        x (array-like): Predictor/input values.
        y (array-like): Target/reference values to fit.

    Returns:
        fitted_a (float): Fitted slope a in y = a*x + b, or np.nan if fitting fails.
        fitted_b (float): Fitted intercept b in y = a*x + b, or np.nan if fitting fails.

    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    fitted_a = float(a)
    fitted_b = float(b)
    return fitted_a, fitted_b

def regression_metrics(y_true, y_pred):
    """
    Compute standard regression metrics between predictions and reference values
    (MAE, RMSE, and R^2). Returns NaNs if there are too few valid samples.

    Args:
        y_true (array-like): Ground-truth / reference values.
        y_pred (array-like): Predicted values.

    Returns:
        dict:
            - "mae" (float): Mean Absolute Error, or np.nan if undefined.
            - "rmse" (float): Root Mean Squared Error, or np.nan if undefined.
            - "r2" (float): Coefficient of determination R^2, or np.nan if undefined.

    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if len(y_true) < 3:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}
