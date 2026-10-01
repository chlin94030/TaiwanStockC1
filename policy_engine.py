"""Trade-plan and entry-state engine for Taiwan Alpha Radar V13.1."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINE_VERSION = "v13.0-policy"
HORIZONS = ["short", "mid", "long"]


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev).abs(),
            (df["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def setup_from_df(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 25:
        return "BASE"
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    p = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else ma20
    atr = float(_true_range(df).rolling(14).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = p * 0.03
    high20_prev = float(pd.to_numeric(df["High"], errors="coerce").shift(1).rolling(20).max().iloc[-1])
    vol20 = float(volume.tail(20).mean()) or 1.0
    vr = float(volume.iloc[-1]) / vol20
    prev_ma20 = float(close.rolling(20).mean().iloc[-2])

    if np.isfinite(high20_prev) and p >= high20_prev * 0.995 and vr >= 1.15:
        return "BREAKOUT"
    if prev < prev_ma20 and p > ma20 and p > ma60:
        return "RECLAIM"
    if p > ma60 and abs(p - ma20) <= 1.1 * atr:
        return "PULLBACK"
    if p > ma5 > ma20 > ma60:
        return "TREND"
    if vr < 0.70 and abs(p / ma20 - 1.0) < 0.04:
        return "DRYUP"
    return "BASE"


def generate_trade_plan(df: pd.DataFrame, horizon: str) -> dict | None:
    if df is None or df.empty or len(df) < 25:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    p = float(close.iloc[-1])
    if not np.isfinite(p) or p <= 0:
        return None

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else ma20
    ma120 = float(close.rolling(120).mean().iloc[-1]) if len(close) >= 120 else ma60
    atr = float(_true_range(df).rolling(14).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = p * 0.03
    high20_prev = float(high.shift(1).rolling(20).max().iloc[-1])
    if not np.isfinite(high20_prev):
        high20_prev = float(high.tail(20).max())

    setup = setup_from_df(df)
    if horizon == "short":
        trigger = max(high20_prev * 1.002, p if setup == "BREAKOUT" else 0.0)
        anchor = ma5 if setup in {"BREAKOUT", "TREND"} else ma20
        zone_low = max(0.01, anchor - 0.65 * atr)
        zone_high = anchor + 0.45 * atr
        chase_limit = max(trigger + 0.85 * atr, p + 0.75 * atr)
        structural = ma20 - 0.75 * atr
        invalidation = max(structural, p - 2.0 * atr)
        entry_mode = "breakout_or_retest"
    elif horizon == "mid":
        trigger = high20_prev * 1.003
        zone_low = max(0.01, ma20 - 0.85 * atr)
        zone_high = ma20 + 0.65 * atr
        chase_limit = max(trigger + 1.15 * atr, p + 1.0 * atr)
        structural = ma60 - 0.65 * atr
        invalidation = max(structural, p - 2.6 * atr)
        entry_mode = "trend_pullback_or_breakout"
    else:
        trigger = high20_prev * 1.005
        anchor = ma60 if p > ma60 else ma120
        zone_low = max(0.01, anchor - 1.05 * atr)
        zone_high = anchor + 0.85 * atr
        chase_limit = max(trigger + 1.60 * atr, p + 1.35 * atr)
        structural = ma120 - 0.80 * atr
        invalidation = max(structural, p - 3.2 * atr)
        entry_mode = "major_trend_accumulation"

    # Never return an invalid stop above current price or inverted zone.
    invalidation = min(invalidation, p * 0.995)
    if zone_low > zone_high:
        zone_low, zone_high = zone_high, zone_low
    return {
        "trigger": round(float(trigger), 2),
        "zone_low": round(float(zone_low), 2),
        "zone_high": round(float(zone_high), 2),
        "chase_limit": round(float(chase_limit), 2),
        "invalidation": round(float(max(0.01, invalidation)), 2),
        "entry_mode": entry_mode,
        "setup": setup,
        "atr14": round(float(atr), 2),
        "ma20": round(float(ma20), 2),
        "ma60": round(float(ma60), 2),
        "ma120": round(float(ma120), 2),
    }


def evaluate_entry_state(df: pd.DataFrame, plan: dict | None) -> str:
    if not plan or df is None or df.empty:
        return "NO_RETURN_ESTIMATE"
    p = float(df["Close"].iloc[-1])
    if not np.isfinite(p):
        return "DATA_UNVERIFIED"
    if p <= float(plan["invalidation"]):
        return "INVALIDATED"
    if p > float(plan["chase_limit"]):
        return "DO_NOT_CHASE"
    if float(plan["zone_low"]) <= p <= float(plan["zone_high"]):
        return "CONDITIONS_MET_NOT_FILLED"
    if p >= float(plan["trigger"]):
        return "CONDITIONS_MET_NOT_FILLED"
    if p < float(plan["zone_low"]):
        return "WAIT_ENTRY_ZONE"
    return "WAIT_BREAKOUT"
