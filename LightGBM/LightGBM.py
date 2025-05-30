"""lgbm_fullyear_forecast.py  —  v2

Fixed LightGBM workflow with categorical‑safe forecast loop
────────────────────────────────────────────────────────────
• trains base booster on 10 reference buildings
• fine‑tunes +500 trees on the new building’s Jan‑Feb data
• prints February MAE / MAPE
• autoregressively predicts Mar‑Dec 2023 and writes one CSV that
  stitches Jan‑Feb actuals with Mar‑Dec forecasts

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ─────────────────────────  PATHS & CONST  ─────────────────────────
DATA_PATH = Path("data/combined_data_preprocessed.csv")
AREAS_PATH = Path("data/areas.csv")
NEW_BUILDING_PATH = Path("data/new_building.csv")
AREA_NEW = 6_991.0

MODEL_DIR = Path("model/lightgbm"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV =  "prediction/new_building_fullyear_2023.csv"

BASE_TREES = 4_000
FINE_TUNE_TREES = 500
WEATHER = ["air temperature", "Atm pressure mm of mercury", "Relative humidity (%)"]
CAL = ["hour", "day_of_week", "month", "is_weekend"]
LAGS = ["lag_1", "lag_24"]
ROLLS = ["roll3", "roll24"]

# ─────────────────────  LOAD & PREP REFERENCE DATA  ────────────────
print("🗄️  loading site‑wide data …")
site = pd.read_csv(DATA_PATH, parse_dates=["Timestamps"], dayfirst=True)
site.columns = site.columns.str.strip().str.replace('"', '')
site.set_index("Timestamps", inplace=True)
site.sort_index(inplace=True)

# calendar
site["hour"] = site.index.hour
site["day_of_week"] = site.index.dayofweek
site["month"] = site.index.month
site["is_weekend"] = (site.index.dayofweek >= 5).astype(int)

# areas
areas = pd.read_csv(AREAS_PATH)
areas.columns = areas.columns.str.strip().str.replace('"', '')
area_map = areas.set_index("Buid_ID")["Area [m2]"].astype(float).to_dict()

BLDG_COLS = [c for c in site.columns if c in area_map]
if not BLDG_COLS:
    raise RuntimeError("No matching building IDs in dataset vs areas.csv")

records = []
for b in BLDG_COLS:
    df = site[[b, *WEATHER, *CAL]].copy()
    df["building_id"] = b
    df["area_m2"] = area_map[b]
    df.rename(columns={b: "kwh"}, inplace=True)
    records.append(df)
long = pd.concat(records)

# engineering target + lags / rolls
long["spec_kwh"] = long["kwh"] / long["area_m2"]
long[LAGS[0]] = long.groupby("building_id")["spec_kwh"].shift(1)
long[LAGS[1]] = long.groupby("building_id")["spec_kwh"].shift(24)
long[ROLLS[0]] = long.groupby("building_id")["spec_kwh"].rolling(3).mean().reset_index(level=0,drop=True)
long[ROLLS[1]] = long.groupby("building_id")["spec_kwh"].rolling(24).mean().reset_index(level=0,drop=True)
long.dropna(inplace=True)

# categorical handling
long["building_id"] = long["building_id"].astype("category").cat.add_categories(["NEW"])
CAT_FEATURES = ["building_id"]

FEATURES = CAT_FEATURES + CAL + WEATHER + LAGS + ROLLS
TARGET = "spec_kwh"

train_ds = lgb.Dataset(long[FEATURES], label=long[TARGET], categorical_feature=CAT_FEATURES)
PARAMS = dict(objective="regression",
              metric="mae",
              learning_rate=0.05,
              num_leaves=256,
              feature_fraction=0.9,
              bagging_fraction=0.8,
              bagging_freq=1,
              verbosity=-1)
print("🏋️  training base booster …")
base = lgb.train(PARAMS, train_ds, num_boost_round=BASE_TREES)
base.save_model(MODEL_DIR / "lgbm_base.txt")

# ───────────────────  PREP NEW BUILDING JAN‑FEB  ────────────────────
print("🔧  fine‑tuning on Jan–Feb …")
nb = pd.read_csv(NEW_BUILDING_PATH, names=["Timestamps", "kwh"], header=0, dayfirst=True)
nb["Timestamps"] = pd.to_datetime(nb["Timestamps"], dayfirst=True)
nb.set_index("Timestamps", inplace=True)
nb = nb.join(site[WEATHER + CAL], how="left")

nb["building_id"] = "NEW"
nb["area_m2"] = AREA_NEW
nb["spec_kwh"] = nb["kwh"] / AREA_NEW
for lag,name in zip([1,24], LAGS):
    nb[name] = nb["spec_kwh"].shift(lag)
nb[ROLLS[0]] = nb["spec_kwh"].rolling(3).mean()
nb[ROLLS[1]] = nb["spec_kwh"].rolling(24).mean()
nb.dropna(inplace=True)
nb["building_id"] = nb["building_id"].astype("category")  # same dtype as training

ft_ds = lgb.Dataset(nb[FEATURES], label=nb[TARGET], categorical_feature=CAT_FEATURES)
booster = lgb.train(PARAMS, ft_ds, num_boost_round=FINE_TUNE_TREES, init_model=base)
booster.save_model(MODEL_DIR / "lgbm_after_feb.txt")

# ─────────────────────  FEBRUARY METRICS  ──────────────────────────
feb = nb.loc["2023-02-01":"2023-02-28 23:00:00"]
pred_spec = booster.predict(feb[FEATURES])
mae = mean_absolute_error(feb["kwh"], pred_spec * AREA_NEW)
mape = mean_absolute_percentage_error(feb["kwh"], pred_spec * AREA_NEW) * 100
print(f"📈  FEB RESULTS — MAE: {mae:,.2f} | MAPE: {mape:,.2f}%")

# ─────────────────────  MAR–DEC FORECAST  ──────────────────────────
print("🔮  forecasting Mar–Dec …")
future_hours = pd.date_range("2023-03-01 00:00:00", "2023-12-31 23:00:00", freq="h")
nb_full = nb.copy()

for ts in future_hours:
    if ts not in nb_full.index:
        # Try to pull weather/time from site; handle gaps (e.g. DST skip) gracefully
        if ts in site.index:
            base_vals = site.loc[ts, WEATHER + CAL]
        else:
            # Hour missing in site dataframe (likely daylight‑saving change);
            # create synthetic calendar row and leave weather as NaN (model handles it)
            base_vals = pd.Series(index=WEATHER + CAL, dtype=float)
            base_vals["hour"] = ts.hour
            base_vals["day_of_week"] = ts.dayofweek
            base_vals["month"] = ts.month
            base_vals["is_weekend"] = int(ts.dayofweek >= 5)
        nb_full.loc[ts, WEATHER + CAL] = base_vals.values
        nb_full.at[ts, "building_id"] = "NEW"
        nb_full.at[ts, "area_m2"] = AREA_NEW
        # placeholder spec for lag calc
        nb_full.at[ts, "spec_kwh"] = np.nan

        # compute lags/rolls using up‑to‑date spec_kwh
        ts_idx = nb_full.index.get_loc(ts)
        nb_full.at[ts, LAGS[0]] = nb_full.iloc[ts_idx-1]["spec_kwh"] if ts_idx>=1 else np.nan
        nb_full.at[ts, LAGS[1]] = nb_full.iloc[ts_idx-24]["spec_kwh"] if ts_idx>=24 else np.nan
        nb_full.at[ts, ROLLS[0]] = nb_full.iloc[max(0,ts_idx-2):ts_idx+1]["spec_kwh"].mean()
        nb_full.at[ts, ROLLS[1]] = nb_full.iloc[max(0,ts_idx-23):ts_idx+1]["spec_kwh"].mean()

        row = nb_full.loc[[ts], FEATURES]
        if row[LAGS+ROLLS].isna().any(axis=None):
            # if not enough history yet (early March), skip prediction this hour
            continue
        spec_pred = booster.predict(row)[0]
        nb_full.at[ts, "spec_kwh"] = spec_pred

nb_full["predicted_kwh"] = nb_full["spec_kwh"] * AREA_NEW
nb_full.loc[:"2023-02-28 23:00:00", "predicted_kwh"] = nb_full.loc[:"2023-02-28 23:00:00", "kwh"].values

nb_full[["predicted_kwh"]].to_csv(OUT_CSV)
print("📊  full‑year CSV →", OUT_CSV)
