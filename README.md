# Macro Forecasting with a Large Number of Predictors

**Author:** Elena Rocco
This project — originally developed as part of my Big Data Analytics coursework at the University of Bologna — is inspired by a forecasting exercise by De Mol, Giannone & Reichlin (2008). It compares eight forecasting models for one-step-ahead prediction of US industrial production and inflation: autoregressive models, Ridge, Lasso, and Principal Component Regression.

A full write-up is in `report/report.pdf`.

---

## Repository Structure
```
macro-forecasting/
├── data/
│   ├── raw/                    # original FRED-MD vintage (Feb 2026)
│   ├── interim/
│   │   ├── interim.py          # data cleaning and stationarity transforms
│   │   └── fredmd_*.csv        # intermediate files produced by interim.py
│   └── processed/              # final datasets used in the notebooks
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis
│   ├── 02_forecast.ipynb       # hyperparameter tuning and forecast
│   └── 03_results.ipynb        # RMSE comparison across models
├── src/
│   ├── transform.py            # standardisation and back-transform helpers
│   ├── models.py               # model tuning, forecasting, RMSE utilities
│   └── plots.py                # plotting helpers
├── references/                 # original papers and assignment roadmap
└── report/
    ├── report.pdf
    └── report.tex 
```

## Data

The analysis uses the FRED-MD dataset, a monthly panel of ~120 US macroeconomic series maintained by the Federal Reserve Bank of St. Louis. You can download it here:  
https://www.stlouisfed.org/research/economists/mccracken/fred-databases

Place the downloaded file in `data/raw/` before running anything.

---

## User Guide

Install dependencies:
```bash
pip install -r requirements.txt
```

Then run the notebooks in order:
```
notebooks/eda.ipynb → notebooks/forecast.ipynb → notebooks/results.ipynb
```

**To replicate with a newer data vintage:** download the latest FRED-MD file from the link above, place it in `data/raw/`, update the filename in the first line of `data/interim/interim.py`:
```python
raw = pd.read_csv('data/raw/2026-02-MD.csv', ...)  # replace with new filename
```

Then run the data pipeline before the notebooks:
```bash
python data/interim/interim.py
```

---

## References

- De Mol, C., Giannone, D., & Reichlin, L. (2008). *Forecasting using a large number of predictors: Is Bayesian shrinkage a valid alternative to principal components?* Journal of Econometrics, 146(2), 318–328.
- McCracken, M. W., & Ng, S. (2016). *FRED-MD: A monthly database for macroeconomic research.* Journal of Business & Economic Statistics, 34(4), 574–589.
