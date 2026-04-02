import sys
import os
from numpy import size
import pandas as pd

# Go up to the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

#import functions
from src.transform import transform_series, remove_outliers, standardize

raw = pd.read_csv('data/raw/2026-02-MD.csv', index_col=0)

#extract tcodes from dataset
tcodes = raw.iloc[0,1:].astype(int).values
data = raw.iloc[1:,:].astype(float)

data.index = pd.to_datetime(data.index, dayfirst=True, format="%m/%d/%Y")
data = data.astype(float)

data.to_csv(os.path.join(os.path.dirname(__file__), "../../data/interim/fredmd_raw.csv"))

#1. Apply transformations and remove outliers

transformed_cols = {
        col: transform_series(data[col], tcodes[i])
        for i, col in enumerate(data.columns[1:])
}
df_transformed = pd.DataFrame(transformed_cols, index=data.index)

df_transformed = df_transformed.apply(remove_outliers)
df_transformed.to_csv(os.path.join(os.path.dirname(__file__), "../../data/interim/fredmd_transformed.csv"))

#2. Manage missing values
#drop columns with more than x% of missing values

threshold = 0.05
missing_cols = df_transformed.isna().sum()

cols_to_drop = missing_cols[missing_cols / len(df_transformed) > threshold].index
df_transformed.drop(columns=cols_to_drop, inplace=True)

#drop oilprice due to instability
df_transformed.drop(columns=["OILPRICEx"], inplace=True, errors="ignore")

#fill remaining missing values with forward fill
df_transformed.ffill(inplace=True)
#remove first months, some series start later
df_transformed=df_transformed.dropna()

#export the cleaned dataset
df_transformed .to_csv(os.path.join(os.path.dirname(__file__), "../../data/interim/fredmd_cleaned.csv"))

#3. delete covid: exogenous shock, not useful for forecasting
df_transformed = df_transformed[(df_transformed.index < pd.Timestamp('2020-01-01')) | (df_transformed.index > pd.Timestamp('2020-06-01'))]
df_transformed.to_csv(os.path.join(os.path.dirname(__file__), "../../data/processed/fredmd_final.csv"))

#4.raw cleaned data

# select dates
raw_clean = data[(data.index >= pd.Timestamp('1960-01-01'))].copy()

#drop cols with more than x% of missing values
threshold = 0.05
missing_cols = raw_clean.isna().sum()
cols_to_drop = missing_cols[missing_cols / len(raw_clean) > threshold].index
raw_clean.drop(columns=cols_to_drop, inplace=True)

#remove remaining NA
raw_clean.ffill(inplace=True)

# remove covid
raw_clean = raw_clean[(raw_clean.index < pd.Timestamp('2020-01-01')) | (raw_clean.index > pd.Timestamp('2020-06-01'))]
raw_clean.to_csv(os.path.join(os.path.dirname(__file__), "../../data/processed/fredmd_raw_cleaned.csv"))