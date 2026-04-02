import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import warnings
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# BIC for RIDGE and LASSO 

def _bic(rss, k, n):
    """
    Standard BIC formula.
    Requires: data standardized (zero mean, unit variance) — no intercept needed.
    
    Parameters
    ----------
    rss : float - sum of squared residuals
    k   : float - effective degrees of freedom
    n   : int   - sample size
    """
    return n * np.log(rss / n) + k * np.log(n)


# ============================================================
# HYPERPARAMETER TUNING
# All functions use the first training window only.
# Optimal parameters are then kept fixed for the entire loop.
# Requires: X and y standardized (zero mean, unit variance).=

def tune_ar(y, p_max=12):
    """
    Select optimal lag order p for AR(p) via BIC.
    Uses statsmodels AutoReg with no intercept (trend='n').
    Requires: y standardized (zero mean, unit variance).
    
    Parameters
    ----------
    y     : array-like - standardized target variable
    p_max : int        - maximum lag order (default: 12)
    
    Returns
    -------
    best_p : int - optimal lag order
    """
    bic_scores = []
    for p in range(1, p_max + 1):
        try:
            model = AutoReg(y, lags=p, trend="n", old_names=False).fit()
            bic_scores.append((model.bic, p))
        except:
            continue

    if not bic_scores:
        raise ValueError("No AR model converged.")
    
    return min(bic_scores, key=lambda x: x[0])[1]



def tune_ridge(X, y, alphas=np.logspace(-6, 6, 200)):
    """
    Select optimal lambda for Ridge via BIC.
    df = sum(d^2 / (d^2 + lambda)) — no intercept term.
    Requires: X and y standardized (zero mean, unit variance).
    
    Parameters
    ----------
    X      : array (n, p) - standardized predictor matrix
    y      : array (n,)   - standardized target variable
    alphas : array        - grid of lambda values (default: 1e-6 to 1e6)
    
    Returns
    -------
    best_alpha : float - optimal lambda
    bic_scores : list  - BIC values for each lambda (for plotting)
    """
    n = len(y)
    U, d, Vt = np.linalg.svd(X, full_matrices=False)

    bic_scores = []
    for alpha in alphas:
        coef = Vt.T @ np.diag(d / (d**2 + alpha)) @ U.T @ y
        rss  = np.sum((y - X @ coef)**2)
        df   = np.sum(d**2 / (d**2 + alpha))
        bic_scores.append(_bic(rss, df, n))

    return alphas[np.argmin(bic_scores)], bic_scores


def tune_lasso(X, y, alphas=np.logspace(-6, 1, 100), suppress_warnings=True):
    """
    Select optimal lambda for Lasso via BIC.
    df = number of non-zero coefficients — no intercept term.
    Requires: X and y standardized (zero mean, unit variance).
    
    Parameters
    ----------
    X      : array (n, p) - standardized predictor matrix
    y      : array (n,)   - standardized target variable
    alphas : array        - grid of lambda values (default: 1e-4 to 10)
    suppress_warnings : bool - whether to suppress convergence warnings (default: True)
    
    Returns
    -------
    best_alpha : float - optimal lambda
    bic_scores : list  - BIC values for each lambda (for plotting)
    """
    n = len(y)
    bic_scores = []

    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False).fit(X, y)
        rss   = np.sum((y - model.predict(X))**2)
        df    = np.sum(model.coef_ != 0)
        bic_scores.append(_bic(rss + 1e-10, df, n))

    return alphas[np.argmin(bic_scores)], bic_scores


def tune_pcr(X):
    """
    Returns explained variance ratios for scree plot.
    Optimal number of components r chosen visually.
    Requires: X standardized (zero mean, unit variance).
    
    Parameters
    ----------
    X : array (n, p) - standardized predictor matrix
    
    Returns
    -------
    explained_variance_ratio : array - explained variance per component
    """
    from sklearn.decomposition import PCA
    pca = PCA().fit(X)
    return pca.explained_variance_ratio_


def rmse(forecast, actual):
    """
    Computes Root Mean Squared Error between forecast and actual values.
    Requires: forecast and actual as arrays of the same length.

    Parameters
    ----------
    forecast : array - model forecasts
    actual   : array - realized values
    """
    return np.sqrt(np.mean((np.array(forecast) - np.array(actual))**2))


def rmse_ci(forecasts, actual, alpha=0.05):
    z = 1.96
    results = {}
    
    for model in forecasts.columns:
        e2 = (forecasts[model].values - actual.values) ** 2
        mse = e2.mean()
        
        # Newey-West variance of mean(e²)
        hac_var = sm.stats.sandwich_covariance.cov_hac(
            sm.OLS(e2, np.ones(len(e2))).fit()
        ).item() / len(e2)
        
        se_rmse = np.sqrt(hac_var / (4 * mse))
        rmse_val = np.sqrt(mse)
        
        results[model] = dict(RMSE=rmse_val, 
                              lower=rmse_val - z * se_rmse, 
                              upper=rmse_val + z * se_rmse)
    
    return pd.DataFrame(results).T.sort_values("RMSE")
    