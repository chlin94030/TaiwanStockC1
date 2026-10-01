"""Offline self-test for Alpha Radar V13.1.  No network access required."""
from __future__ import annotations

import tempfile
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# The execution sandbox used for packaging may not have yfinance installed.
# market_data only needs it for live downloads, which this offline test never calls.
try:
    import yfinance  # noqa: F401
except Exception:
    yf_stub = types.ModuleType("yfinance")
    def _offline_only(*args, **kwargs):
        raise RuntimeError("yfinance live access is not used by offline self-test")
    yf_stub.download = _offline_only
    yf_stub.Ticker = _offline_only
    sys.modules["yfinance"] = yf_stub

from market_data import ResearchDataClient, _yf_history_args
from policy_engine import generate_trade_plan, evaluate_entry_state
from return_first_model import estimate_horizon_return
from radar_service import _fundamental_score, _flow_score, _combined_ranking_score
from industry_profile import fine_industry
from presentation import investment_view


def synthetic_prices(seed=42, n=1250):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-10-01", periods=n)
    rets = rng.normal(0.00065, 0.018, n)
    close = 100 * np.exp(np.cumsum(rets))
    op = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(op, close) * (1 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(op, close) * (1 - rng.uniform(0.001, 0.018, n))
    vol = rng.lognormal(np.log(2_000_000), 0.35, n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


class FakeClient(ResearchDataClient):
    def _finmind(self, dataset, stock_id, start_date, end_date=None, ttl_hours=6.0):
        if dataset == "TaiwanStockMonthRevenue":
            rows = []
            base = pd.Period("2024-09", freq="M")
            for i in range(25):
                p = base + i
                rows.append({
                    "date": p.start_time.strftime("%Y-%m-%d"), "stock_id": stock_id, "country": "Taiwan",
                    "revenue": 10_000_000_000 * (1 + i * 0.02), "revenue_month": p.month, "revenue_year": p.year,
                })
            return rows
        if dataset == "TaiwanStockFinancialStatements":
            rows = []
            vals = [
                ("2025-12-31", 2.0, 100e9, 40e9), ("2026-03-31", 2.4, 105e9, 43e9),
                ("2026-06-30", 2.8, 112e9, 47e9), ("2026-09-30", 3.1, 120e9, 51e9),
            ]
            for d, eps, rev, gp in vals:
                for typ, val in [("EPS", eps), ("Revenue", rev), ("GrossProfit", gp)]:
                    rows.append({"date": d, "stock_id": stock_id, "type": typ, "value": val, "origin_name": typ})
            return rows
        if dataset == "TaiwanStockPER":
            return [{"date": "2026-09-30", "stock_id": stock_id, "PER": "24.5", "PBR": "4.2", "dividend_yield": "2.1"}]
        if dataset == "TaiwanStockInstitutionalInvestorsBuySellWide":
            return [{
                "date": f"2026-09-{i+1:02d}", "stock_id": stock_id,
                "Foreign_Investor_buy": 2_000_000, "Foreign_Investor_sell": 1_500_000,
                "Foreign_Dealer_Self_buy": 0, "Foreign_Dealer_Self_sell": 0,
                "Investment_Trust_buy": 500_000, "Investment_Trust_sell": 350_000,
                "Dealer_buy": 300_000, "Dealer_sell": 250_000,
            } for i in range(20)]
        return []


def main():
    assert "start" in _yf_history_args("8y")
    assert _yf_history_args("5y") == {"period": "5y"}

    # Fine-industry labels are deterministic and intentionally plain-language.
    assert fine_industry("2330.TW", "台積電", "半導體業") == "半導體－晶圓代工龍頭"
    assert fine_industry("2317.TW", "鴻海", "其他電子業") == "電子中游－EMS/垂直整合"
    assert fine_industry("2308.TW", "台達電", "電子零組件業") == "電子中游－電源與冷卻系統"
    assert fine_industry("3017.TW", "奇鋐", "電腦及週邊設備業") == "電子中游－水冷散熱"

    df = synthetic_prices()
    settings = SimpleNamespace(commission=0.001425, sell_tax=0.003, slippage=0.0005)
    for h in ["short", "mid", "long"]:
        f = estimate_horizon_return(df, h, settings, twii_ret_20d=0.02)
        p = generate_trade_plan(df, h)
        assert f.get("estimate_available") is True
        assert 0 <= f.get("composite_factor_score", -1) <= 100
        assert f.get("local_effective_n", 0) >= 30
        assert p["invalidation"] < float(df["Close"].iloc[-1])
        assert p["zone_low"] <= p["zone_high"]
        assert evaluate_entry_state(df, p) in {
            "CONDITIONS_MET_NOT_FILLED", "WAIT_ENTRY_ZONE", "WAIT_BREAKOUT", "DO_NOT_CHASE", "INVALIDATED"
        }

    with tempfile.TemporaryDirectory() as td:
        client = FakeClient(Path(td) / "research.sqlite")
        r = client.stock_research("2330.TW")
        assert len(r["monthly_revenue"]["rows"]) == 12
        assert abs(r["financials"]["eps_4q_sum"] - 10.3) < 1e-9
        assert r["financials"]["quarters"][-1]["eps_ytd_sum_unadjusted"] == 8.3
        assert r["valuation"]["pe"] == 24.5
        assert r["institutional_flow"]["total_net_lots"] > 0
        fundamental, _ = _fundamental_score(r)
        flow, _ = _flow_score(r, 2_000_000)
        assert fundamental is not None and flow is not None

        stock = {
            "ticker": "2330.TW", "name": "台積電", "industry": "半導體業",
            "fine_industry": fine_industry("2330.TW", "台積電", "半導體業"),
            "avg_volume_20d": 2_000_000, "research": r, "horizons": {}
        }
        for h in ["short", "mid", "long"]:
            stock["horizons"][h] = {
                "entry_state": "CONDITIONS_MET_NOT_FILLED",
                "forecast": {
                    "estimate_available": True, "technical_factor_score": 70, "composite_factor_score": 70,
                    "alpha_mean": 0.03, "local_effective_n": 80,
                    "strategy": {"median": 0.04, "mean": 0.05, "p10": -0.04,
                                 "expected_shortfall10_loss": -0.07, "historical_positive_rate": 0.62},
                },
            }
            _combined_ranking_score(stock, h, "full")
            assert 0 <= stock["horizons"][h]["ranking_score"] <= 100
            view = investment_view(stock, h, {"regime": "BULL"})
            assert view["title"]
            assert len(view["reasons"]) >= 1
            assert "半導體－晶圓代工龍頭" in view["summary"]

    print("Alpha Radar V13.1 offline self-test: PASS")


if __name__ == "__main__":
    main()
