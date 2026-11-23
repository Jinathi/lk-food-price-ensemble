import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
SEED = 42
START = "2023-01-01"      # dense, post-crisis window of the WFP data
MIN_OBS = 24              # keep market-commodity series with at least this many months
MIN_HIST = 8              # months of history ARIMA needs before it forecasts
NUM = ["lp1", "d1", "d2", "mrev", "ln1", "nd1", "prem", "msin", "mcos", "lat", "lon"]
CAT = ["commodity", "market", "category"]
def load_clean(path):
    """Retail prices since 2023, one row per market x commodity x month."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df.pricetype == "Retail") & (df.date >= START) & (df.price > 0)].copy()
    df["month"] = df.date.dt.to_period("M")
    return df.groupby(["market", "commodity", "month"], as_index=False).agg(
        price=("price", "mean"), admin1=("admin1", "first"),
        category=("category", "first"), unit=("unit", "first"), lat=("latitude", "first"),
        lon=("longitude", "first"))
def build_panel(g, min_obs=MIN_OBS, min_hist=MIN_HIST):
    """Returns one row per series x month, including one future month (for forecasting).

    Features for month t use observations from months < t only. No price is
    interpolated from future observations. An observed prior-month price is
    required for every evaluated one-month-ahead forecast.
    """
    last = g.month.max()
    months = pd.period_range(g.month.min(), last + 1, freq="M")   # + one future month
    n_obs = months[:-1]
    counts = g.groupby(["market", "commodity"]).month.nunique()
    keep = counts[counts >= min_obs].index
    rows = []
    for (mk, cm), s in g.groupby(["market", "commodity"]):
        if (mk, cm) not in keep:
            continue
        p = s.set_index("month").price.reindex(n_obs)
        # Keep missing months missing: interpolation between past and future
        # prices would leak future information into the lagged features.
        lp = np.log(p)
        lp = lp.reindex(months)
        meta = s.iloc[0]
        rows.append(pd.DataFrame({
            "market": mk, "commodity": cm, "category": meta.category,
            "unit": meta.unit, "admin1": meta.admin1,
            "lat": meta.lat, "lon": meta.lon,
            "month": months, "t": np.arange(len(months)),
            "price": p.reindex(months).values, "lp": lp.values}))
    d = pd.concat(rows, ignore_index=True).sort_values(["market", "commodity", "t"])
    d["obs"] = d.price.notna()

    # own-price lags
    grp = d.groupby(["market", "commodity"]).lp
    for k in (1, 2, 3):
        d[f"lp{k}"] = grp.shift(k)

    # National (cross-market) average uses OBSERVED prices only. In particular,
    # it never interpolates a missing historical value using a later month.
    nat = d.groupby(["commodity", "t"]).lp.mean().rename("nat").reset_index()
    nat = nat.sort_values(["commodity", "t"])
    nat["ln1"] = nat.groupby("commodity").nat.shift(1)
    nat["ln2"] = nat.groupby("commodity").nat.shift(2)

    # rolling one-step ARIMA(1,1,0) forecast of the national log price
    T = len(months)
    arima = []
    for cm, s in nat.groupby("commodity"):
        series = s.nat.values[: T - 1]
        series = pd.Series(series).ffill()  # only information available by each date
        for t in range(min_hist, T):
            history = series.iloc[:t].dropna().to_numpy()
            if len(history) < min_hist:
                continue
            try:
                fc = ARIMA(history, order=(1, 1, 0)).fit().forecast(1)[0]
            except Exception:
                fc = history[-1]
            arima.append((cm, t, fc - history[-1]))
    arima = pd.DataFrame(arima, columns=["commodity", "t", "arima_dchg"])
    d = d.merge(nat[["commodity", "t", "ln1", "ln2"]], on=["commodity", "t"], how="left")
    d = d.merge(arima, on=["commodity", "t"], how="left")
    d["dy"] = d.lp - d.lp1                         # target: monthly log price change
    # A valid one-month change needs the immediately preceding observed month.
    # Missing lag months are excluded, rather than filled from later observations.
    d = d.dropna(subset=["lp1", "lp2", "lp3", "ln1", "ln2", "arima_dchg"])
    return derive(d).reset_index(drop=True)


def derive(d):
    """Derived features. Also used by the dashboard when the user edits lag prices."""
    d = d.copy()
    d["d1"] = d.lp1 - d.lp2                       # last month's change (momentum)
    d["d2"] = d.lp2 - d.lp3
    d["mrev"] = (d.lp1 + d.lp2 + d.lp3) / 3 - d.lp1   # distance from 3-month mean
    d["nd1"] = d.ln1 - d.ln2                      # national momentum
    d["prem"] = d.lp1 - d.ln1                     # market premium over national price
    m = d.month.dt.month
    d["msin"] = np.sin(2 * np.pi * m / 12)
    d["mcos"] = np.cos(2 * np.pi * m / 12)
    return d


def get_categories(panel):
    return {c: sorted(panel[c].unique()) for c in CAT}


def encode(df, cats):
    """One-hot encoding with fixed columns (so train, test and dashboard rows align)."""
    parts = [df[NUM].reset_index(drop=True)]
    for c in CAT:
        cc = pd.Categorical(df[c].values, categories=cats[c])
        parts.append(pd.get_dummies(cc, prefix=c).astype(float))
    return pd.concat(parts, axis=1)
class Model:
    """Wraps one regressor so every model takes the same panel DataFrame and predicts dy."""

    def __init__(self, kind, cats):
        self.kind, self.cats = kind, cats
        self.est = self._make(kind)

    def _make(self, kind):
        if kind == "Random Forest":      # bagging
            return RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5,
                                         n_jobs=-1, random_state=SEED)
        if kind == "GBM":                # boosting
            return GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, random_state=SEED)
        if kind == "XGBoost":            # boosting
            return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
        if kind == "CatBoost":           # boosting with native categorical handling
            return CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                                     random_seed=SEED, verbose=0, thread_count=-1)
        if kind == "Ridge":              # linear reference model
            return make_pipeline(StandardScaler(), Ridge(alpha=10))
        if kind == "ANN":                # artificial neural network
            return make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=500, random_state=SEED))
        raise ValueError(kind)

    def _X(self, df):
        if self.kind == "CatBoost":
            return df[NUM + CAT].reset_index(drop=True)
        return encode(df, self.cats).values

    def fit(self, df, y):
        if self.kind == "CatBoost":
            self.est.fit(self._X(df), y, cat_features=CAT)
        else:
            self.est.fit(self._X(df), y)
        return self

    def predict(self, df):
        return self.est.predict(self._X(df))


class Stack:
    """Stacking: base models -> out-of-fold predictions -> meta-model (non-negative Ridge).

    Out-of-fold predictions use forward-chaining over time (train on the past, predict the
    next block) so the meta-model never learns from the future.
    """

    def __init__(self, cats, bases=("Random Forest", "XGBoost", "CatBoost", "Ridge")):
        self.cats, self.bases = cats, bases

    def fit(self, df, y):
        y = np.asarray(y)
        ts = np.sort(df.t.unique())
        cuts = [ts[int(len(ts) * q)] for q in (0.35, 0.55, 0.78)] + [ts.max() + 1]
        oof = np.full((len(df), len(self.bases)), np.nan)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            tr, va = (df.t < lo).values, ((df.t >= lo) & (df.t < hi)).values
            for j, b in enumerate(self.bases):
                m = Model(b, self.cats).fit(df[tr], y[tr])
                oof[va, j] = m.predict(df[va])
        ok = ~np.isnan(oof).any(axis=1)
        self.meta = Ridge(alpha=1.0, positive=True).fit(oof[ok], y[ok])
        self.models = [Model(b, self.cats).fit(df, y) for b in self.bases]
        return self

    def predict(self, df):
        P = np.column_stack([m.predict(df) for m in self.models])
        return self.meta.predict(P)

    @property
    def weights(self):
        return dict(zip(self.bases, self.meta.coef_.round(3)))


class ArimaRF:
    """Hybrid RF + ARIMA: ARIMA gives the trend, a Random Forest learns what ARIMA misses."""

    def __init__(self, cats):
        self.rf = Model("Random Forest", cats)

    def fit(self, df, y):
        self.rf.fit(df, np.asarray(y) - df.arima_dchg.values)
        return self

    def predict(self, df):
        return df.arima_dchg.values + self.rf.predict(df)


class AvgHybrid:
    """Hybrid RF + ANN: average of a Random Forest and a neural network."""

    def __init__(self, cats, kinds=("Random Forest", "ANN")):
        self.ms = [Model(k, cats) for k in kinds]

    def fit(self, df, y):
        for m in self.ms:
            m.fit(df, y)
        return self

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.ms], axis=0)


class Naive:
    """Baseline: next month's price = this month's price (predicted change = 0)."""
    def fit(self, df, y): return self
    def predict(self, df): return np.zeros(len(df))


class ArimaOnly:
    """Baseline: apply ARIMA's predicted national change to every market."""
    def fit(self, df, y): return self
    def predict(self, df): return df.arima_dchg.values


def build_all(cats):
    return {
        "Naive (last month)": Naive(),
        "ARIMA only": ArimaOnly(),
        "Ridge (linear)": Model("Ridge", cats),
        "Random Forest": Model("Random Forest", cats),
        "GBM": Model("GBM", cats),
        "XGBoost": Model("XGBoost", cats),
        "CatBoost": Model("CatBoost", cats),
        "Stacking": Stack(cats),
        "Hybrid RF+ARIMA": ArimaRF(cats),
        "Hybrid RF+ANN": AvgHybrid(cats),
    }


FAMILY = {"Naive (last month)": "Baseline", "ARIMA only": "Baseline", "Ridge (linear)": "Baseline",
          "Random Forest": "Bagging", "GBM": "Boosting", "XGBoost": "Boosting",
          "CatBoost": "Boosting", "Stacking": "Stacking",
          "Hybrid RF+ARIMA": "Hybrid", "Hybrid RF+ANN": "Hybrid"}
def to_price(df, dy_hat):
    """Convert a predicted log change back to LKR."""
    return np.exp(df.lp1.values + dy_hat)


def metrics(df, dy_hat):
    y, p = df.price.values, to_price(df, dy_hat)
    return {"MAE (LKR)": mean_absolute_error(y, p),
            "RMSE (LKR)": np.sqrt(mean_squared_error(y, p)),
            "MAPE (%)": np.mean(np.abs(y - p) / y) * 100,
            "R2": r2_score(y, p),
            "Direction acc. (%)": (np.nan if np.allclose(dy_hat, 0) else
                                  np.mean(np.sign(dy_hat) == np.sign(df.dy.values)) * 100)}
