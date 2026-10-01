"""
Taiwan Alpha Radar V13.1 - empirical return / technical factor engine.

Design goals
------------
1. Rank-first: always produce a relative ranking when usable price data exists.
2. No fabricated forecast: insufficient evidence returns estimate_available=False.
3. Return estimates are empirical distributions from historically similar states,
   not compounded extrapolations of the latest momentum.
4. Separate "factor score" from "confidence/evidence" and from realized-return
   statistics.  The confidence score is NOT a win probability.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


class ModelDataError(Exception):
    pass


def finite_scalar(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def _feature_frame(df: pd.DataFrame, twii_ret_20d: float = 0.0) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    x = df[list(required)].copy().sort_index()
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Close"])
    if x.empty:
        return pd.DataFrame()

    close = x["Close"]
    vol = x["Volume"].fillna(0.0)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma240 = close.rolling(240).mean()

    atr14 = _true_range(x).rolling(14).mean()
    vol20 = vol.rolling(20).mean()
    high20_prev = x["High"].shift(1).rolling(20).max()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)

    f = pd.DataFrame(index=x.index)
    f["ret5"] = close.pct_change(5)
    f["ret20"] = close.pct_change(20)
    f["ret60"] = close.pct_change(60)
    f["ret120"] = close.pct_change(120)
    f["gap20"] = close / ma20 - 1.0
    f["gap60"] = close / ma60 - 1.0
    f["gap120"] = close / ma120 - 1.0
    f["gap240"] = close / ma240 - 1.0
    f["ma20_slope"] = ma20.pct_change(10)
    f["ma60_slope"] = ma60.pct_change(20)
    f["vol_ratio"] = vol / vol20.replace(0, np.nan)
    f["atr_pct"] = atr14 / close.replace(0, np.nan)
    f["rsi14"] = _rsi(close) / 100.0
    f["macd_pct"] = macd_hist / close.replace(0, np.nan)
    f["bb_pos"] = ((close - bb_lower) / bb_width).clip(-0.5, 1.5)
    f["breakout20"] = (close / high20_prev.replace(0, np.nan) - 1.0).clip(-0.30, 0.30)
    f["rs20"] = f["ret20"] - float(twii_ret_20d or 0.0)

    # rolling downside / trend stability features
    daily = close.pct_change()
    f["volatility20"] = daily.rolling(20).std()
    f["drawdown60"] = close / close.rolling(60).max() - 1.0

    # diagnostic-only raw series
    f["close"] = close
    f["ma5"] = ma5
    f["ma20"] = ma20
    f["ma60"] = ma60
    f["ma120"] = ma120
    f["ma240"] = ma240
    f["atr14"] = atr14
    return f.replace([np.inf, -np.inf], np.nan)


def _technical_score(row: pd.Series, horizon: str) -> float:
    """Bounded 0-100 cross-feature score.  No single hard gate."""
    r5 = finite_scalar(row.get("ret5"))
    r20 = finite_scalar(row.get("ret20"))
    r60 = finite_scalar(row.get("ret60"))
    r120 = finite_scalar(row.get("ret120"))
    rs20 = finite_scalar(row.get("rs20"))
    vr = finite_scalar(row.get("vol_ratio"), 1.0)
    g20 = finite_scalar(row.get("gap20"))
    g60 = finite_scalar(row.get("gap60"))
    g120 = finite_scalar(row.get("gap120"))
    slope20 = finite_scalar(row.get("ma20_slope"))
    slope60 = finite_scalar(row.get("ma60_slope"))
    rsi = finite_scalar(row.get("rsi14"), 0.5)
    macd = finite_scalar(row.get("macd_pct"))
    breakout = finite_scalar(row.get("breakout20"))
    atr = finite_scalar(row.get("atr_pct"), 0.03)
    dd60 = finite_scalar(row.get("drawdown60"), -0.05)

    def s(centered, lo, hi):
        return float(_clip01((centered - lo) / max(1e-9, hi - lo)))

    # Reward positive structure while avoiding pure overextension chasing.
    trend = (
        0.28 * s(g20, -0.05, 0.08)
        + 0.24 * s(g60, -0.08, 0.16)
        + 0.20 * s(g120, -0.12, 0.25)
        + 0.16 * s(slope20, -0.03, 0.08)
        + 0.12 * s(slope60, -0.05, 0.12)
    )
    relative = s(rs20, -0.08, 0.12)
    volume = s(vr, 0.65, 1.80)
    breakout_s = s(breakout, -0.06, 0.035)
    macd_s = s(macd, -0.01, 0.015)
    momentum_short = 0.45 * s(r5, -0.06, 0.10) + 0.55 * s(r20, -0.10, 0.22)
    momentum_mid = 0.35 * s(r20, -0.12, 0.25) + 0.65 * s(r60, -0.18, 0.45)
    momentum_long = 0.25 * s(r60, -0.22, 0.55) + 0.75 * s(r120, -0.30, 0.80)

    # 45-72 RSI is usually constructive; very high RSI receives no extra benefit.
    rsi_quality = max(0.0, 1.0 - abs(rsi - 0.60) / 0.35)
    risk_quality = 0.65 * (1.0 - s(atr, 0.02, 0.085)) + 0.35 * s(dd60, -0.30, -0.02)

    if horizon == "short":
        score = (
            0.23 * momentum_short + 0.20 * relative + 0.16 * volume + 0.13 * breakout_s
            + 0.12 * trend + 0.08 * macd_s + 0.04 * rsi_quality + 0.04 * risk_quality
        )
    elif horizon == "mid":
        score = (
            0.24 * trend + 0.21 * momentum_mid + 0.18 * relative + 0.11 * volume
            + 0.09 * breakout_s + 0.07 * macd_s + 0.05 * rsi_quality + 0.05 * risk_quality
        )
    else:
        score = (
            0.34 * trend + 0.24 * momentum_long + 0.14 * relative + 0.05 * volume
            + 0.05 * breakout_s + 0.05 * macd_s + 0.05 * rsi_quality + 0.08 * risk_quality
        )
    return round(float(np.clip(score * 100.0, 0.0, 100.0)), 1)


def _historical_analog_returns(
    features: pd.DataFrame,
    horizon_days: int,
    total_cost: float,
    current_row: pd.Series,
) -> dict:
    cols = [
        "ret5", "ret20", "ret60", "gap20", "gap60", "ma20_slope", "vol_ratio",
        "atr_pct", "rsi14", "macd_pct", "bb_pos", "breakout20", "drawdown60",
    ]
    hist = features[cols + ["close"]].copy()
    hist["fwd"] = hist["close"].shift(-horizon_days) / hist["close"] - 1.0 - total_cost
    # Exclude the latest observation and any row without a forward outcome.
    hist = hist.iloc[:-max(1, horizon_days)].dropna(subset=cols + ["fwd"])
    if len(hist) < 35:
        return {"available": False, "n": len(hist)}

    cur = current_row.reindex(cols).astype(float)
    if cur.isna().any():
        return {"available": False, "n": len(hist)}

    X = hist[cols].astype(float)
    med = X.median()
    mad = (X - med).abs().median().replace(0, np.nan)
    # Fallback scale to std, then 1 to prevent division by zero.
    std = X.std(ddof=0).replace(0, np.nan)
    scale = (mad * 1.4826).fillna(std).fillna(1.0).clip(lower=1e-6)
    z_hist = (X - med) / scale
    z_cur = (cur - med) / scale

    weights = pd.Series(
        [1.1, 1.4, 1.0, 1.25, 1.1, 0.9, 0.7, 0.8, 0.55, 0.55, 0.45, 0.9, 0.7],
        index=cols,
    )
    dist = np.sqrt(((z_hist - z_cur) ** 2 * weights).sum(axis=1) / weights.sum())
    k = int(np.clip(round(len(hist) * 0.12), 40, 140))
    nearest_idx = dist.nsmallest(k).index
    sample = hist.loc[nearest_idx, "fwd"].replace([np.inf, -np.inf], np.nan).dropna()
    nearest_dist = dist.loc[nearest_idx].dropna()
    if len(sample) < 30:
        return {"available": False, "n": len(sample)}

    q10 = float(sample.quantile(0.10))
    lower = sample[sample <= q10]
    es10 = float(lower.mean()) if not lower.empty else q10
    mean = float(sample.mean())
    median = float(sample.median())
    p75 = float(sample.quantile(0.75))
    win_rate = float((sample > 0).mean())
    dispersion = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0

    # Evidence quality: sample size + similarity + sign agreement. Not probability.
    n_score = min(1.0, len(sample) / 90.0)
    similarity = float(np.exp(-max(0.0, float(nearest_dist.median())) / 2.2)) if not nearest_dist.empty else 0.0
    sign_strength = min(1.0, abs(win_rate - 0.5) * 2.0)
    stability = 1.0 - min(1.0, dispersion / max(0.04, abs(median) * 4.0 + 0.04))
    evidence_score = 100.0 * (0.35 * n_score + 0.35 * similarity + 0.20 * sign_strength + 0.10 * stability)

    return {
        "available": True,
        "n": int(len(sample)),
        "mean": mean,
        "median": median,
        "p75": p75,
        "p10": q10,
        "expected_shortfall10_loss": es10,
        "win_rate": win_rate,
        "dispersion": dispersion,
        "median_distance": float(nearest_dist.median()) if not nearest_dist.empty else math.nan,
        "evidence_score": float(np.clip(evidence_score, 20.0, 98.0)),
    }


def estimate_horizon_return(df: pd.DataFrame, horizon: str, settings, twii_ret_20d: float = 0.005) -> dict:
    if df is None or df.empty or len(df) < 80:
        return {"estimate_available": False, "sample_supported": False, "reason": "PRICE_HISTORY_TOO_SHORT"}

    horizon_days = {"short": 10, "mid": 40, "long": 120}.get(horizon, 40)
    features = _feature_frame(df, twii_ret_20d=twii_ret_20d)
    if features.empty:
        return {"estimate_available": False, "sample_supported": False, "reason": "FEATURES_UNAVAILABLE"}

    current = features.iloc[-1]
    factor_score = _technical_score(current, horizon)
    total_cost = (
        finite_scalar(getattr(settings, "commission", 0.001425)) * 2.0
        + finite_scalar(getattr(settings, "sell_tax", 0.003))
        + finite_scalar(getattr(settings, "slippage", 0.0005)) * 2.0
    )
    analog = _historical_analog_returns(features, horizon_days, total_cost, current)

    rs20 = finite_scalar(current.get("rs20"))
    base = {
        "estimate_available": bool(analog.get("available", False)),
        "sample_supported": bool(analog.get("available", False)),
        "is_outperforming_market": rs20 > 0,
        "composite_factor_score": factor_score,
        "technical_factor_score": factor_score,
        "alpha_mean": round(rs20, 4),  # current 20d relative strength, not a forecasted alpha
        "features": {
            "ret5": round(finite_scalar(current.get("ret5")), 4),
            "ret20": round(finite_scalar(current.get("ret20")), 4),
            "ret60": round(finite_scalar(current.get("ret60")), 4),
            "vol_ratio": round(finite_scalar(current.get("vol_ratio"), 1.0), 3),
            "atr_pct": round(finite_scalar(current.get("atr_pct")), 4),
            "rsi14": round(finite_scalar(current.get("rsi14"), 0.5) * 100.0, 1),
            "gap20": round(finite_scalar(current.get("gap20")), 4),
            "gap60": round(finite_scalar(current.get("gap60")), 4),
        },
    }

    if not analog.get("available"):
        base.update({
            "confidence_score": 0.0,
            "strategy": {},
            "local_effective_n": float(analog.get("n", 0)),
            "local_time_blocks": 0,
            "local_weight": 0.0,
            "reason": "ANALOG_SAMPLE_TOO_SMALL",
        })
        return base

    base.update({
        "confidence_score": round(finite_scalar(analog.get("evidence_score")), 1),
        "strategy": {
            "mean": round(finite_scalar(analog.get("mean")), 4),
            "median": round(finite_scalar(analog.get("median")), 4),
            "p75": round(finite_scalar(analog.get("p75")), 4),
            "p10": round(finite_scalar(analog.get("p10")), 4),
            "expected_shortfall10_loss": round(finite_scalar(analog.get("expected_shortfall10_loss")), 4),
            "historical_positive_rate": round(finite_scalar(analog.get("win_rate")), 4),
            "dispersion": round(finite_scalar(analog.get("dispersion")), 4),
        },
        "local_effective_n": float(analog.get("n", 0)),
        "local_time_blocks": int(max(1, analog.get("n", 0) // 20)),
        "local_weight": round(float(np.clip(analog.get("n", 0) / 100.0, 0.30, 0.95)), 2),
        "analog_median_distance": round(finite_scalar(analog.get("median_distance")), 3),
    })
    return base


def holding_review(price: float, original_invalidation=None, trailing_protection=None, thesis_broken=None, prices_verified=True) -> str:
    if not prices_verified:
        return "DATA_UNVERIFIED"
    if original_invalidation and price <= original_invalidation:
        return "ORIGINAL_STRUCTURE_INVALIDATED"
    if thesis_broken is True:
        return "ORIGINAL_THESIS_INVALIDATED"
    if trailing_protection and price <= trailing_protection:
        return "PROTECTION_TRIGGER_REVIEW_EXECUTION"
    if not original_invalidation and not trailing_protection:
        return "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW"
    return "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE"
