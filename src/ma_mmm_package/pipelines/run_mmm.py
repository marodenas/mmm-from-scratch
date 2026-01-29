from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


# =============================
# Config
# =============================
@dataclass
class MMMConfig:
    # columns
    date_col: str = "DATE_DAY"
    org_col: str = "ORGANISATION_ID"
    territory_col: str = "TERRITORY_NAME"
    target_col: str = "ALL_PURCHASES"

    # channel detection
    spend_suffix: str = "_SPEND"
    spend_cols: Optional[List[str]] = None  # if None, auto-detect by suffix

    # aggregation
    freq: str = "W-MON"  # weekly anchored to Monday

    # adstock
    theta: float = 0.5

    # saturation (Hill)
    sat_alpha: float = 1.0
    k_quantile: float = 0.5
    k_eps: float = 1e-8

    # ridge
    ridge_alpha: float = 1.0

    # time validation
    min_train_weeks: int = 104
    test_weeks: int = 26
    step_weeks: int = 26

    # time controls
    add_trend: bool = True
    add_seasonality: bool = True
    seasonality_period: int = 52
    seasonality_order: int = 2


# =============================
# Utilities
# =============================
def detect_spend_cols(df: pd.DataFrame, suffix: str) -> List[str]:
    return [c for c in df.columns if c.endswith(suffix)]


def to_weekly(
    df: pd.DataFrame,
    cfg: MMMConfig,
    org_id: str,
    territory_name: str
) -> pd.DataFrame:
    """Filter to one series (org + territory) and aggregate to weekly."""
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])

    d = d[(d[cfg.org_col] == org_id) & (d[cfg.territory_col] == territory_name)].copy()
    if d.empty:
        raise ValueError("No rows after filtering. Check org_id / territory_name.")

    spend_cols = cfg.spend_cols or detect_spend_cols(d, cfg.spend_suffix)
    keep_cols = [cfg.date_col, cfg.target_col] + spend_cols
    d = d[keep_cols].copy()

    # Weekly aggregation
    d = (
        d.set_index(cfg.date_col)
         .sort_index()
         .resample(cfg.freq)
         .sum(min_count=1)
         .reset_index()
         .rename(columns={cfg.date_col: "week"})
    )

    # Fill missing spends with 0 (no spend weeks)
    d[spend_cols] = d[spend_cols].fillna(0.0)

    # Target: choose a policy. Here: fill missing with 0 for continuity
    d[cfg.target_col] = d[cfg.target_col].fillna(0.0)

    return d


# =============================
# Feature engineering
# =============================
def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Geometric adstock: recursive carryover."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i in range(len(x)):
        carry = x[i] + theta * carry
        out[i] = carry
    return out


def hill_saturation(x: np.ndarray, alpha: float, k: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x ** alpha) / (x ** alpha + k ** alpha)


def safe_k_from_series(x: pd.Series, q: float = 0.5, eps: float = 1e-8) -> float:
    """Robust k: quantile of positive values only; eps fallback."""
    x_pos = x[x > 0]
    if len(x_pos) == 0:
        return eps
    k = float(x_pos.quantile(q))
    return max(k, eps)


def make_trend_feature(n: int) -> pd.DataFrame:
    """Simple linear trend t = 0..n-1."""
    return pd.DataFrame({"trend_t": np.arange(n, dtype=float)})


def make_fourier_seasonality(dates: np.ndarray, period: int = 52, order: int = 2) -> pd.DataFrame:
    """
    Fourier seasonality features for weekly data.
    Uses week-of-year (ISO week) mapped onto [0, 2*pi].
    """
    dt = pd.to_datetime(dates)
    week_of_year = dt.isocalendar().week.astype(int).to_numpy()

    t = 2.0 * np.pi * (week_of_year / float(period))

    feats = {}
    for k in range(1, order + 1):
        feats[f"sin_{period}_k{k}"] = np.sin(k * t)
        feats[f"cos_{period}_k{k}"] = np.cos(k * t)

    return pd.DataFrame(feats)


def build_features(
    df_weekly: pd.DataFrame,
    cfg: MMMConfig,
    spend_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build:
      - X_ads: adstocked spend
      - X_final: time controls (optional) + adstock+saturation features
      - y, dates
      - k_by_channel
    """
    dates = pd.to_datetime(df_weekly["week"]).values
    y = df_weekly[cfg.target_col].astype(float).values

    # --- Adstock ---
    X_ads = pd.DataFrame(index=df_weekly.index)
    for col in spend_cols:
        X_ads[col] = geometric_adstock(df_weekly[col].values, theta=cfg.theta)

    # --- Saturation ---
    k_by_channel: Dict[str, float] = {}
    X_sat = pd.DataFrame(index=df_weekly.index)
    for col in spend_cols:
        k = safe_k_from_series(X_ads[col], q=cfg.k_quantile, eps=cfg.k_eps)
        k_by_channel[col] = k
        X_sat[col] = hill_saturation(X_ads[col].values, alpha=cfg.sat_alpha, k=k)

    # --- Time controls ---
    time_parts = []
    if cfg.add_trend:
        time_parts.append(make_trend_feature(len(df_weekly)))
    if cfg.add_seasonality:
        time_parts.append(
            make_fourier_seasonality(
                dates=dates,
                period=cfg.seasonality_period,
                order=cfg.seasonality_order
            )
        )

    if len(time_parts) > 0:
        X_time = pd.concat(time_parts, axis=1)
        X_final = pd.concat([X_time.reset_index(drop=True), X_sat.reset_index(drop=True)], axis=1)
    else:
        X_final = X_sat.copy()

    return X_ads, X_final, y, dates, k_by_channel


# =============================
# Model + backtesting
# =============================
def make_ridge_model(ridge_alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ridge_alpha))
    ])


def rolling_origin_splits(n: int, min_train_weeks: int, test_weeks: int, step_weeks: int):
    splits = []
    train_end = min_train_weeks
    while train_end + test_weeks <= n:
        tr = np.arange(0, train_end)
        te = np.arange(train_end, train_end + test_weeks)
        splits.append((tr, te))
        train_end += step_weeks
    return splits


def backtest_time_series(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: np.ndarray,
    cfg: MMMConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # sort by date
    order = np.argsort(dates)
    X = X.iloc[order].reset_index(drop=True)
    y = np.asarray(y).ravel()[order]
    dates = dates[order]

    splits = rolling_origin_splits(
        n=len(X),
        min_train_weeks=cfg.min_train_weeks,
        test_weeks=cfg.test_weeks,
        step_weeks=cfg.step_weeks,
    )
    if len(splits) == 0:
        raise ValueError("No splits created. Reduce min_train_weeks or test_weeks.")

    rows = []
    coef_rows = []

    for i, (tr, te) in enumerate(splits, start=1):
        model = make_ridge_model(cfg.ridge_alpha)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])

        rows.append({
            "fold": i,
            "train_start": pd.to_datetime(dates[tr][0]).date(),
            "train_end": pd.to_datetime(dates[tr][-1]).date(),
            "test_start": pd.to_datetime(dates[te][0]).date(),
            "test_end": pd.to_datetime(dates[te][-1]).date(),
            "r2_test": r2_score(y[te], pred),
            "mae_test": mean_absolute_error(y[te], pred),
        })

        coef = model.named_steps["ridge"].coef_
        coef_rows.append(pd.Series(coef, index=X.columns, name=f"fold_{i}"))

    results = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)

    return results, coefs


# =============================
# End-to-end runner
# =============================
def run_mmm_for_series(
    df: pd.DataFrame,
    cfg: MMMConfig,
    org_id: str,
    territory_name: str,
) -> Dict[str, object]:
    # 1) Weekly table
    df_weekly = to_weekly(df, cfg, org_id, territory_name)

    # 2) Spend cols
    spend_cols = cfg.spend_cols or detect_spend_cols(df_weekly, cfg.spend_suffix)

    # 3) Features
    X_ads, X_final, y, dates, k_by_channel = build_features(df_weekly, cfg, spend_cols)

    # 4) Backtest
    results, coef_folds = backtest_time_series(X_final, y, dates, cfg)

    # 5) Summaries
    coef_summary = pd.DataFrame({
        "mean": coef_folds.mean(),
        "std": coef_folds.std(),
        "cv": coef_folds.std() / (coef_folds.mean().abs() + 1e-8),
    }).sort_values("cv")

    return {
        "df_weekly": df_weekly,
        "spend_cols": spend_cols,
        "X_ads": X_ads,
        "X_final": X_final,
        "k_by_channel": k_by_channel,
        "results": results,
        "coef_folds": coef_folds,
        "coef_summary": coef_summary,
    }
