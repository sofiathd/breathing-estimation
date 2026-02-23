import numpy as np
from src.vt_calib.linear import regression_metrics, fit_linear_calibration

def blocked_kfold_indices(n, k=5):
    """
    Splits indices into k contiguous chunks.

    Args:
        n (int): Total number of samples.
        k (int): Number of folds to create.

    Returns:
        list[np.ndarray]: List of length k, where each element is an array of indices for
        one contiguous fold.
    """
    k = int(max(2, k))
    idx = np.arange(n)
    out = np.array_split(idx, k)
    return out

def cv_calibrate_linear_blocked(x, y, k=5, min_train=20, min_test=5):
    """
    Evaluate a linear calibration model (y=ax+b) using blocked K-fold CV.

    The data is split into k contiguous folds. Each fold is used once as the test set, while all remaining samples form the training set. For each valid split, a and b are fitted on training data and evaluated on the held-out fold using regression metrics.

    Args:
        x (array-like): Input values to calibrate (predictor variable).
        y (array-like): Target/reference values (response variable).
        k (int): Number of contiguous folds to use.
        min_train (int): Minimum number of training samples required for a fold to be used.
        min_test (int): Minimum number of test samples required for a fold to be used.

    Returns:
        dict: Cross-validation summary metrics:
            - "mae" (float): Mean absolute error averaged across used folds (or NaN).
            - "rmse" (float): Root mean squared error averaged across used folds (or NaN).
            - "r2" (float): R² averaged across used folds (or NaN).
            - "k_used" (int): Number of folds actually used (after filtering).
            - "n" (int): Number of valid (finite) samples used after cleaning.

    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < (min_train + min_test):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "k_used": 0, "n": n}

    folds = blocked_kfold_indices(n, k=k)

    maes, rmses, r2s = [], [], []
    k_used = 0

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if len(train_idx) < min_train or len(test_idx) < min_test:
            continue

        a, b = fit_linear_calibration(x[train_idx], y[train_idx])
        if not np.isfinite(a) or not np.isfinite(b):
            continue

        y_hat = a * x[test_idx] + b
        met = regression_metrics(y[test_idx], y_hat)
        maes.append(met["mae"]); rmses.append(met["rmse"]); r2s.append(met["r2"])
        k_used += 1

    if k_used == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "k_used": 0, "n": n}

    return {
        "mae": float(np.nanmean(maes)),
        "rmse": float(np.nanmean(rmses)),
        "r2": float(np.nanmean(r2s)),
        "k_used": int(k_used),
        "n": int(n),
    }

