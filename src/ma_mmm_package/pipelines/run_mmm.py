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
    """
    Detect spend columns by suffix.

    Parameters
    ----------
    df:
        Input dataframe.
    suffix:
        Suffix used to identify spend columns (e.g., "_SPEND").

    Returns
    -------
    list[str]
        Column names ending with `suffix`.

    Notes
    -----
    This is a convenience helper. In production, you may prefer an explicit
    allowlist to avoid accidentally including unexpected columns.
    """
    return [c for c in df.columns if c.endswith(suffix)]

    return [c for c in df.columns if c.endswith(suffix)]


def to_weekly(
    df: pd.DataFrame,
    cfg: MMMConfig,
    org_id: str,
    territory_name: str
) -> pd.DataFrame:
    """
    Filter the raw MMM dataset to a single time series and aggregate to weekly.

    This function:
    1) Filters rows by (organisation_id, territory_name)
    2) Keeps date, target, and spend columns
    3) Aggregates to weekly granularity using `cfg.freq`
    4) Fills missing spends with 0.0 and missing target with 0.0

    Parameters
    ----------
    df:
        Raw dataset containing multiple organisations/territories and daily data.
    cfg:
        MMM configuration (column names, frequency, spend suffix, etc.).
    org_id:
        Organisation identifier (one series).
    territory_name:
        Territory selection (one series), e.g. "All Territories".

    Returns
    -------
    pd.DataFrame
        Weekly dataframe with columns:
        - "week" (datetime)
        - cfg.target_col
        - spend columns (auto-detected or cfg.spend_cols)

    Raises
    ------
    ValueError
        If filtering by org_id and territory_name results in an empty dataframe.

    Notes
    -----
    - Weekly aggregation uses SUM. This matches typical MMM practice for spends
      and count-like outcomes. If your target is an average-rate KPI, you may
      need a different aggregation (mean/weighted mean).
    - Filling target NAs with 0.0 is a modeling choice for continuity; in some
      contexts you may prefer to drop missing target weeks.
    """
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])

    d = d[(d[cfg.org_col] == org_id) & (d[cfg.territory_col] == territory_name)].copy()
    if d.empty:
        raise ValueError("No rows after filtering. Check org_id / territory_name.")

    spend_cols = cfg.spend_cols or detect_spend_cols(d, cfg.spend_suffix)
    keep_cols = [cfg.date_col, cfg.target_col] + spend_cols
    d = d[keep_cols].copy()

    d = (
        d.set_index(cfg.date_col)
         .sort_index()
         .resample(cfg.freq)
         .sum(min_count=1)
         .reset_index()
         .rename(columns={cfg.date_col: "week"})
    )

    d[spend_cols] = d[spend_cols].fillna(0.0)
    d[cfg.target_col] = d[cfg.target_col].fillna(0.0)

    return d


# =============================
# Feature engineering
# =============================
def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply geometric adstock (recursive carryover) to a spend series.

    The geometric adstock transformation is:

    .. math::
        \\tilde{x}_t = x_t + \\theta \\tilde{x}_{t-1}

    where:
    - :math:`x_t` is the raw spend at time t
    - :math:`\\tilde{x}_t` is the adstocked spend
    - :math:`\\theta \\in [0, 1]` controls carryover memory

    Parameters
    ----------
    x:
        1D array of spend values ordered in time.
    theta:
        Carryover parameter. Higher values imply longer persistence.

    Returns
    -------
    np.ndarray
        Adstocked series of the same shape as `x`.

    Notes
    -----
    - This is a simple carryover model. Other frameworks may use alternative
      adstock kernels (e.g., Weibull) with more flexible shapes.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i in range(len(x)):
        carry = x[i] + theta * carry
        out[i] = carry
    return out


def hill_saturation(x: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """
    Apply Hill saturation (diminishing returns) to an input series.

    The Hill function is:

    .. math::
        f(x) = \\frac{x^{\\alpha}}{x^{\\alpha} + k^{\\alpha}}

    where:
    - :math:`k` is the half-saturation point (50% of max effect)
    - :math:`\\alpha` controls curve steepness

    Parameters
    ----------
    x:
        1D array (typically adstocked spend) with non-negative values.
    alpha:
        Shape/steepness parameter (alpha > 0).
    k:
        Half-saturation parameter (k > 0).

    Returns
    -------
    np.ndarray
        Saturated values in (0, 1) for x > 0. If x == 0 => 0.

    Notes
    -----
    This transformation is applied per-channel to model diminishing marginal
    returns at higher spend levels.
    """
    x = np.asarray(x, dtype=float)
    return (x ** alpha) / (x ** alpha + k ** alpha)


def safe_k_from_series(x: pd.Series, q: float = 0.5, eps: float = 1e-8) -> float:
    """
    Compute a robust half-saturation parameter k from a series.

    By default, k is set as a quantile of strictly positive values:

    - If the series has positive values: k = quantile(x_pos, q)
    - If no positive values exist: k = eps

    Parameters
    ----------
    x:
        Input series (e.g., adstocked spend for one channel).
    q:
        Quantile to use (default 0.5 = median).
    eps:
        Small positive fallback to avoid division by zero.

    Returns
    -------
    float
        Robust k value (>= eps).

    Notes
    -----
    This is a pragmatic heuristic for "from scratch" MMM. In production MMM,
    k is typically estimated/optimized (often Bayesian or via hyperparameter search).
    """
    x_pos = x[x > 0]
    if len(x_pos) == 0:
        return eps
    k = float(x_pos.quantile(q))
    return max(k, eps)


def make_trend_feature(n: int) -> pd.DataFrame:
    """
    Create a simple linear trend feature for weekly time series.

    Parameters
    ----------
    n:
        Number of rows (weeks) in the time series.

    Returns
    -------
    pd.DataFrame
        Single-column dataframe with:
        - trend_t = [0, 1, ..., n-1]

    Notes
    -----
    The trend coefficient is learned by the regression model. A positive
    coefficient implies upward baseline drift; negative implies decline.
    """
    return pd.DataFrame({"trend_t": np.arange(n, dtype=float)})


def make_fourier_seasonality(dates: np.ndarray, period: int = 52, order: int = 2) -> pd.DataFrame:
    """
    Create Fourier seasonality features for weekly time series.

    Uses ISO week-of-year to generate sin/cos basis functions:

    .. math::
        \\sin\\left(\\frac{2\\pi k t}{P}\\right),\\ \\cos\\left(\\frac{2\\pi k t}{P}\\right)

    Parameters
    ----------
    dates:
        Array of weekly dates.
    period:
        Seasonal period (default 52 weeks for yearly seasonality).
    order:
        Number of Fourier harmonics. Higher values increase flexibility.

    Returns
    -------
    pd.DataFrame
        Fourier terms with columns:
        - sin_{period}_k1, cos_{period}_k1, ..., sin_{period}_kK, cos_{period}_kK

    Notes
    -----
    - Fourier terms model smooth seasonality and typically generalize better
      than monthly/weekly dummy variables.
    - ISO week-of-year can include week 53 in some years. With period=52,
      those weeks are mapped onto the same angular scale; this is usually fine
      for exploratory MMM work.
    """
    dt = pd.to_datetime(dates)
    woy = dt.isocalendar().week.astype(int).to_numpy()
    t = 2.0 * np.pi * (woy / float(period))

    feats = {}
    for k in range(1, order + 1):
        feats[f"sin_{period}_k{k}"] = np.sin(k * t)
        feats[f"cos_{period}_k{k}"] = np.cos(k * t)

    return pd.DataFrame(feats)


def build_features(
    df_weekly: pd.DataFrame,
    cfg: MMMConfig,
    spend_cols: List[str]
) -> Dict[str, object]:
    """
    Build the MMM design matrix and intermediate feature tables.

    This function constructs, in order:
    1) Adstocked spends (X_ads)
    2) Saturated spends (X_sat) using Hill saturation
    3) Optional time controls (X_time): trend + Fourier seasonality
    4) Final design matrix X_final = [X_time, X_sat] (or just X_sat)

    Parameters
    ----------
    df_weekly:
        Weekly dataframe output of `to_weekly()`. Must include "week", target, and spend columns.
    cfg:
        MMM configuration controlling adstock, saturation, and time controls.
    spend_cols:
        List of spend columns (channel inputs).

    Returns
    -------
    dict
        Dictionary containing:
        - df_weekly: sorted/cleaned weekly table
        - dates: np.ndarray of weekly datetime values
        - y: np.ndarray target
        - spend_cols: list of spend column names
        - time_cols: list of time-control columns
        - channel_cols: list of marketing columns (same names as spend_cols)
        - X_ads: DataFrame (adstocked spend)
        - X_sat: DataFrame (adstock + saturation)
        - X_time: DataFrame (trend + seasonality)
        - X_final: DataFrame (model matrix)
        - k_by_channel: dict[channel -> k]

    Notes
    -----
    - This function is deterministic given cfg and data.
    - In more advanced MMM, theta/k/alpha may be estimated rather than fixed.
    """
    dfw = df_weekly.sort_values("week").reset_index(drop=True).copy()
    dates = pd.to_datetime(dfw["week"]).values
    y = dfw[cfg.target_col].astype(float).values

    # --- Adstock ---
    X_ads = pd.DataFrame(index=dfw.index)
    for col in spend_cols:
        X_ads[col] = geometric_adstock(dfw[col].values, theta=cfg.theta)

    # --- Saturation ---
    k_by_channel: Dict[str, float] = {}
    X_sat = pd.DataFrame(index=dfw.index)
    for col in spend_cols:
        k = safe_k_from_series(X_ads[col], q=cfg.k_quantile, eps=cfg.k_eps)
        k_by_channel[col] = k
        X_sat[col] = hill_saturation(X_ads[col].values, alpha=cfg.sat_alpha, k=k)

    # --- Time controls ---
    time_parts = []
    if cfg.add_trend:
        time_parts.append(make_trend_feature(len(dfw)))
    if cfg.add_seasonality:
        time_parts.append(
            make_fourier_seasonality(
                dates=dates,
                period=cfg.seasonality_period,
                order=cfg.seasonality_order
            )
        )

    if len(time_parts) > 0:
        X_time = pd.concat(time_parts, axis=1).reset_index(drop=True)
        X_final = pd.concat([X_time, X_sat.reset_index(drop=True)], axis=1)
    else:
        X_time = pd.DataFrame(index=dfw.index)
        X_final = X_sat.copy()

    time_cols = list(X_time.columns)
    channel_cols = spend_cols[:]  # X_sat uses same names as spend_cols

    return {
        "df_weekly": dfw,
        "dates": dates,
        "y": y,
        "spend_cols": spend_cols,
        "time_cols": time_cols,
        "channel_cols": channel_cols,
        "X_ads": X_ads,
        "X_sat": X_sat,
        "X_time": X_time,
        "X_final": X_final,
        "k_by_channel": k_by_channel,
    }


# =============================
# Model helpers
# =============================
def make_ridge_model(ridge_alpha: float) -> Pipeline:
    """
    Create the standard modeling pipeline used in this project.

    The pipeline is:
    - StandardScaler: normalize features for stable regularization
    - Ridge: L2-regularized linear regression

    Parameters
    ----------
    ridge_alpha:
        Ridge penalty strength (lambda). Higher values shrink coefficients more.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline ready for fitting and prediction.

    Notes
    -----
    Scaling is important because Ridge regularization depends on feature scale.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ridge_alpha))
    ])


def backtransform_ridge_pipeline(model: Pipeline, feature_names: List[str]) -> Tuple[pd.Series, float]:
    """
    Back-transform Ridge coefficients from standardized feature space to original scale.

    When using StandardScaler, Ridge is trained on:

    .. math::
        z_j = (x_j - \\mu_j) / \\sigma_j

    If the model learns coefficients :math:`\\beta^{scaled}_j` and intercept :math:`b_0`,
    then coefficients on the original feature scale are:

    .. math::
        \\beta^{raw}_j = \\beta^{scaled}_j / \\sigma_j

    and the raw intercept is:

    .. math::
        b_0^{raw} = b_0 - \\sum_j \\beta^{scaled}_j \\mu_j / \\sigma_j

    Parameters
    ----------
    model:
        Fitted Pipeline(StandardScaler, Ridge).
    feature_names:
        List of feature names in the same order as the design matrix.

    Returns
    -------
    (pd.Series, float)
        - beta_raw: coefficients in original feature units
        - intercept_raw: intercept in original target units

    Notes
    -----
    Use raw coefficients for interpretability (e.g., trend units, KPI units).
    """
    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]

    beta_scaled = ridge.coef_.ravel()
    mu = scaler.mean_
    sigma = scaler.scale_ + 1e-12

    beta_raw = beta_scaled / sigma
    intercept_raw = float(ridge.intercept_ - np.sum(beta_scaled * mu / sigma))

    return pd.Series(beta_raw, index=feature_names, name="beta_raw"), intercept_raw


def rolling_origin_splits(n: int, min_train_weeks: int, test_weeks: int, step_weeks: int):
    """
    Generate rolling-origin (walk-forward) train/test splits for time-series validation.

    Parameters
    ----------
    n:
        Number of observations (weeks).
    min_train_weeks:
        Minimum number of weeks in the first training window.
    test_weeks:
        Size of each test window.
    step_weeks:
        Step size to move the training end forward each fold.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_idx, test_idx) index arrays.

    Notes
    -----
    This mimics real forecasting conditions: train on the past, test on the future.
    """
    splits = []
    train_end = min_train_weeks
    while train_end + test_weeks <= n:
        tr = np.arange(0, train_end)
        te = np.arange(train_end, train_end + test_weeks)
        splits.append((tr, te))
        train_end += step_weeks
    return splits


# =============================
# Backtesting + coefficient stability
# =============================
def backtest_time_series(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: np.ndarray,
    cfg: MMMConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform rolling-origin time-series backtesting and track coefficient stability.

    Parameters
    ----------
    X:
        Design matrix (already contains time controls + marketing features).
    y:
        Target array aligned with X.
    dates:
        Date array aligned with X (used for sorting and reporting fold windows).
    cfg:
        MMM configuration controlling splits and ridge strength.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        - results: per-fold performance metrics (R2, MAE) and fold date ranges
        - coefs_scaled: per-fold coefficients in standardized feature space
        - coefs_raw: per-fold coefficients in original feature units (+ __intercept__)

    Notes
    -----
    - Negative out-of-sample R² can occur in MMM due to structural breaks, omitted variables,
      and the fact that MMM is not primarily a forecasting model.
    - Use coefs_raw for interpretation and decomposition; coefs_scaled is mostly diagnostic.
    """
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
    coef_rows_scaled = []
    coef_rows_raw = []

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

        ridge = model.named_steps["ridge"]
        coef_rows_scaled.append(pd.Series(ridge.coef_.ravel(), index=X.columns, name=f"fold_{i}"))

        beta_raw, intercept_raw = backtransform_ridge_pipeline(model, list(X.columns))
        beta_raw.name = f"fold_{i}"
        beta_raw["__intercept__"] = intercept_raw
        coef_rows_raw.append(beta_raw)

    results = pd.DataFrame(rows)
    coefs_scaled = pd.DataFrame(coef_rows_scaled)
    coefs_raw = pd.DataFrame(coef_rows_raw)

    return results, coefs_scaled, coefs_raw


# =============================
# Decomposition on full data
# =============================
def fit_full_model_and_decompose(
    X_final: pd.DataFrame,
    y: np.ndarray,
    time_cols: List[str],
    channel_cols: List[str],
    ridge_alpha: float
) -> Dict[str, object]:
    """
    Fit a single model on the full dataset and compute a decomposition.

    This is intended for interpretability and plotting (not validation):
    - y_hat: model prediction
    - baseline: intercept + (trend + seasonality contribution)
    - marketing_total: total contribution of all marketing channels
    - residual: y - y_hat

    Parameters
    ----------
    X_final:
        Full design matrix.
    y:
        Target array aligned with X_final.
    time_cols:
        Columns representing time controls (trend + seasonality terms).
    channel_cols:
        Columns representing marketing channels.
    ridge_alpha:
        Ridge penalty strength.

    Returns
    -------
    dict
        Contains:
        - model: fitted pipeline
        - beta_raw: back-transformed coefficients in original scale
        - intercept_raw: back-transformed intercept
        - y_hat: predicted KPI
        - baseline: baseline component (intercept + time controls)
        - marketing_total: sum of marketing contributions
        - residual: unexplained component

    Notes
    -----
    The decomposition is:

    y_hat_t = intercept + baseline_time_t + marketing_t

    Baseline here refers to structural dynamics, not “no-marketing counterfactual causality”.
    """
    model = make_ridge_model(ridge_alpha)
    model.fit(X_final, y)

    beta_raw, intercept_raw = backtransform_ridge_pipeline(model, list(X_final.columns))

    # predictions using raw betas (equivalent)
    y_hat = intercept_raw + X_final.values @ beta_raw.values

    # baseline = intercept + time-part
    baseline = np.full(len(X_final), intercept_raw, dtype=float)
    if len(time_cols) > 0:
        baseline += X_final[time_cols].values @ beta_raw[time_cols].values

    # marketing_total = sum over channel cols
    marketing_total = np.zeros(len(X_final), dtype=float)
    if len(channel_cols) > 0:
        marketing_total += X_final[channel_cols].values @ beta_raw[channel_cols].values

    residual = y - y_hat

    return {
        "model": model,
        "beta_raw": beta_raw,
        "intercept_raw": intercept_raw,
        "y_hat": y_hat,
        "baseline": baseline,
        "marketing_total": marketing_total,
        "residual": residual,
    }


# =============================
# End-to-end runner
# =============================
def run_mmm_for_series(
    df: pd.DataFrame,
    cfg: MMMConfig,
    org_id: str,
    territory_name: str,
) -> Dict[str, object]:
    """
    End-to-end MMM runner for a single (organisation_id, territory) time series.

    Pipeline:
    1) Filter + aggregate to weekly
    2) Build features:
       - adstock (carryover)
       - saturation (diminishing returns)
       - optional time controls (trend + Fourier seasonality)
    3) Rolling-origin backtest to evaluate temporal robustness
    4) Fit full model for interpretability and decomposition plots

    Parameters
    ----------
    df:
        Raw dataset containing multiple organisations/territories.
    cfg:
        MMM configuration.
    org_id:
        Organisation identifier to select one series.
    territory_name:
        Territory selection, e.g. "All Territories".

    Returns
    -------
    dict
        Key outputs:
        - df_weekly: weekly table
        - X_ads, X_sat, X_time, X_final: intermediate and final feature tables
        - results: backtest fold metrics
        - coef_folds_raw: fold coefficients (interpretable)
        - coef_summary: coefficient stability summary
        - beta_raw_full, intercept_raw_full: full-fit coefficients (interpretable)
        - baseline_full: intercept + time controls contribution
        - marketing_total_full: sum of channel contributions
        - y_hat_full: model prediction
        - residual_full: y - y_hat_full

    Typical usage
    -------------
    >>> import pandas as pd
    >>> from ma_mmm_package.pipelines.run_mmm import MMMConfig, run_mmm_for_series
    >>> df = pd.read_csv("data/raw/your_dataset.csv")
    >>> cfg = MMMConfig(target_col="ALL_PURCHASES", theta=0.5, add_trend=True, add_seasonality=True)
    >>> out = run_mmm_for_series(df, cfg, org_id="...", territory_name="All Territories")
    >>> out["results"].head()

    Notes
    -----
    - Use `baseline_full` vs observed KPI to understand structural patterns.
    - Use `marketing_total_full` to see incremental deviations explained by media.
    """
    # 1) Weekly table
    df_weekly = to_weekly(df, cfg, org_id, territory_name)

    # 2) Spend cols
    spend_cols = cfg.spend_cols or detect_spend_cols(df_weekly, cfg.spend_suffix)

    # 3) Features
    feats = build_features(df_weekly, cfg, spend_cols)

    # 4) Backtest (scaled + raw coefficients)
    results, coef_folds_scaled, coef_folds_raw = backtest_time_series(
        feats["X_final"], feats["y"], feats["dates"], cfg
    )

    # 5) Coef stability summary (use RAW for interpretability)
    coef_summary = pd.DataFrame({
        "mean": coef_folds_raw.drop(columns=["__intercept__"], errors="ignore").mean(),
        "std": coef_folds_raw.drop(columns=["__intercept__"], errors="ignore").std(),
    })
    coef_summary["cv"] = coef_summary["std"] / (coef_summary["mean"].abs() + 1e-8)
    coef_summary = coef_summary.sort_values("cv")

    # 6) Full-fit decomposition (for plots & narrative, not validation)
    decomp = fit_full_model_and_decompose(
        X_final=feats["X_final"],
        y=feats["y"],
        time_cols=feats["time_cols"],
        channel_cols=feats["channel_cols"],
        ridge_alpha=cfg.ridge_alpha,
    )

    return {
        # data
        "df_weekly": feats["df_weekly"],
        "dates": feats["dates"],
        "y": feats["y"],
        "spend_cols": feats["spend_cols"],

        # features
        "X_ads": feats["X_ads"],
        "X_sat": feats["X_sat"],
        "X_time": feats["X_time"],
        "X_final": feats["X_final"],
        "k_by_channel": feats["k_by_channel"],
        "time_cols": feats["time_cols"],
        "channel_cols": feats["channel_cols"],

        # validation
        "results": results,
        "coef_folds_scaled": coef_folds_scaled,
        "coef_folds_raw": coef_folds_raw,
        "coef_summary": coef_summary,

        # decomposition (full fit)
        "beta_raw_full": decomp["beta_raw"],
        "intercept_raw_full": decomp["intercept_raw"],
        "y_hat_full": decomp["y_hat"],
        "baseline_full": decomp["baseline"],
        "marketing_total_full": decomp["marketing_total"],
        "residual_full": decomp["residual"],
        "model_full": decomp["model"],
    }
