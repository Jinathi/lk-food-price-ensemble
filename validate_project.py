"""Quick checks for the saved CM4371 notebook/dashboard artefacts.

Run from the project directory: python validate_project.py
This is an integrity check, not a substitute for rerunning the notebook.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
res = pd.read_csv(MODELS / 'results.csv', index_col=0)
preds = pd.read_csv(MODELS / 'test_predictions.csv.gz')
panel = pd.read_csv(MODELS / 'panel.csv.gz')
summary = json.loads((MODELS / 'summary.json').read_text())
bundle = joblib.load(MODELS / 'models.joblib')

assert len(bundle['models']) == 10, 'Expected 10 baseline/ensemble models'
assert set(res.index) == set(bundle['models']), 'Results/models mismatch'
assert 'unit' in preds.columns and 'unit' in panel.columns, 'Missing original commodity price units'
assert set(preds.month) == {f'2025-{m:02d}' for m in range(4, 10)}
assert np.isfinite(preds['price']).all()
for name in bundle['models']:
    mae = mean_absolute_error(preds['price'], preds[name])
    assert abs(mae - res.loc[name, 'MAE (LKR)']) < 1e-6, (name, mae)
assert summary['best'] == res['MAE (LKR)'].idxmin()
naive = preds['Naive (last month)']
weighted_gain = np.mean(np.abs(preds.price - naive) - np.abs(preds.price - preds[summary['best']]))
assert abs(weighted_gain - summary['gain']) < 1e-6

# No gaps masquerading as a last-month observation. For every modelled row,
# the lag must match a real price in the immediately preceding calendar month.
panel['month'] = pd.PeriodIndex(panel['month'], freq='M')
panel['previous_month'] = panel['month'] - 1
raw = pd.read_csv(ROOT / 'data' / 'wfp_food_prices_lka.csv')
raw['date'] = pd.to_datetime(raw['date'])
raw = raw[(raw['date'] >= '2023-01-01') & (raw.pricetype == 'Retail') & (raw.price > 0)].copy()
raw['month'] = raw.date.dt.to_period('M')
observed = raw.groupby(['market', 'commodity', 'month'], as_index=False).price.mean()
chk = panel.merge(observed, left_on=['market', 'commodity', 'previous_month'],
                  right_on=['market', 'commodity', 'month'], how='left',suffixes=('','_observed'))
assert chk['price_observed'].notna().all(), 'Forecast row uses a missing previous-month price'
assert np.allclose(np.exp(chk['lp1']), chk['price_observed'], rtol=1e-10, atol=1e-10)

# The explorer only forecasts the true month following the final source observation.
future_month = panel.month.max()
future = panel[panel.month == future_month].copy()
assert future_month == raw.month.max() + 1
assert len(future) > 0
import foodprice as fp
for name, model in bundle['models'].items():
    predicted = fp.to_price(future.head(3), model.predict(future.head(3)))
    assert np.isfinite(predicted).all() and (predicted > 0).all(), name

print(f"PASS: {len(res)} models; {len(preds)} held-out predictions; "
      f"{len(future)} eligible historical future-series forecasts; "
      f"best by MAE: {summary['best']}.")
