import numpy as np
import pandas as pd

# transformations: (1) no transformation; (2) ∆xt; (3) ∆2xt; (4) log(xt); (5) ∆ log(xt); (6) ∆^2(xt). (7) ∆(xt/xt−1 − 1.0). 

def transform_series(x, tcode):
    if tcode == 1: return x                          # no transformation
    if tcode == 2: return x.diff()                   # Δxt
    if tcode == 3: return x.diff().diff()            # Δ²xt
    if tcode == 4: return np.log(x)                  # log(xt)
    if tcode == 5: return np.log(x).diff()           # Δlog (xt)
    if tcode == 6: return np.log(x).diff().diff()    # Δ²log(xt)
    if tcode == 7: return x.pct_change().diff()      # Δ(xt/xt−1 − 1)

#An outlier deviates more than 10 IQR (interquantile ranges) from the median.
def remove_outliers(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 10 * iqr
    upper_bound = q3 + 10 * iqr
    return x[(x >= lower_bound) & (x <= upper_bound)]

#standardize column, return df, mean, and sd for later inverse transformation
def standardize(x):
    """
    Standardize array column-wise (zero mean, unit variance).
    Works for both 1D (vectors) and 2D (matrices) inputs.
    Requires: numpy array.
    """
    mean = x.mean(axis=0)
    std  = x.std(axis=0)
    return (x - mean) / std, mean, std


def rescale_ipi_forecast(forecast_standardized, raw_ipi, last_raw_date_idx):
    """
    IPI: w_t = Δ log IP_t = hat log IP_t - log IP_{t-1}
    Formula: IP_{T+1|T} = exp(log IP_T + ŵ_{T+1|T})
    
    Args:
        forecast_standardized: scalar, il forecast in scala standardizzata
        raw_ipi: array, serie originale (livelli, non trasformata)
        last_raw_date_idx: int, indice nel raw data del periodo T
    
    Returns:
        float: forecast rescalato in scala originale
    """
    log_ip_t = np.log(raw_ipi[last_raw_date_idx])
    log_ip_forecast = log_ip_t + forecast_standardized
    return np.exp(log_ip_forecast)


def rescale_cpi_forecast(forecast_standardized, raw_cpi, last_raw_date_idx):
    """
    CPI: w_t = Δπ_t con π_t = Δ log CPI_t (doppia differenza di log)
    Formula: log CPI_{T+1|T} = ŵ_{T+1|T} + 2 log CPI_T - log CPI_{T-1}
    
    Args:
        forecast_standardized: scalar, il forecast in scala standardizzata
        raw_cpi: array, serie originale (livelli, non trasformata)
        last_raw_date_idx: int, indice nel raw data del periodo T
    
    Returns:
        float: forecast rescalato in scala log (prendi exp se vuoi il livello)
    """
    log_cpi_t = np.log(raw_cpi[last_raw_date_idx])
    log_cpi_t_minus_1 = np.log(raw_cpi[last_raw_date_idx - 1])
    log_cpi_forecast = forecast_standardized + 2 * log_cpi_t - log_cpi_t_minus_1
    return log_cpi_forecast
