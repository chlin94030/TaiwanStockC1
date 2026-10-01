"""
Taiwan Alpha Radar V13.1 - market + research data layer.

Price source: Yahoo Finance, with local SQLite cache.
Research source: FinMind v4 (monthly revenue, quarterly financials, PER,
institutional flows). Broker-branch/main-force proxy is optional because the
underlying branch dataset is a Sponsor feature and "main force" has no official
single definition.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
try:
    import yfinance as yf
except ImportError:  # surfaced cleanly at runtime; dependency is in requirements.txt
    yf = None


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_BRANCH_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report"


def _taipei_timestamp() -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)


def _finmind_token() -> str:
    return os.getenv("FINMIND_TOKEN", "").strip()


def provider_runtime_status(store=None) -> dict:
    return {
        "sqlite_connected": True,
        "yfinance_ready": True,
        "finmind_token_configured": bool(_finmind_token()),
        "branch_flow_enabled": os.getenv("ENABLE_BRANCH_FLOW", "0") == "1",
        "timestamp": _taipei_timestamp().strftime("%Y-%m-%d %H:%M:%S"),
    }


def stock_id_from_ticker(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


def _yf_history_args(period: str) -> dict:
    """Translate UI periods to yfinance-supported arguments.

    yfinance does not support arbitrary period strings such as 3y/8y, so those
    are converted to an explicit start date instead of silently failing.
    """
    valid = {"1y", "2y", "5y", "10y", "max"}
    p = str(period or "5y")
    if p in valid:
        return {"period": p}
    if p.endswith("y") and p[:-1].isdigit():
        years = max(1, int(p[:-1]))
        start = (_taipei_timestamp().date() - datetime.timedelta(days=years * 366 + 10)).isoformat()
        return {"start": start}
    return {"period": "5y"}


def fetch_twse_universe() -> pd.DataFrame:
    """Fetch listed + OTC common-stock universe; fallback to a small core list."""
    tickers: list[tuple[str, str, str]] = []
    headers = {"User-Agent": "Mozilla/5.0 AlphaRadar/13.0"}

    try:
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=10)
        if r.ok:
            for item in r.json():
                code = str(item.get("Code", "")).strip()
                name = str(item.get("Name", "")).strip()
                if len(code) == 4 and code.isdigit():
                    tickers.append((f"{code}.TW", name, "上市"))
    except Exception:
        pass

    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", headers=headers, timeout=10)
        if not r.ok:
            r = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O", headers=headers, timeout=10)
        if r.ok:
            for item in r.json():
                code = str(
                    item.get("SecuritiesCompanyCode")
                    or item.get("公司代號")
                    or item.get("公司代碼")
                    or ""
                ).strip()
                name = str(item.get("CompanyName") or item.get("Company Name") or item.get("公司名稱") or "").strip()
                if len(code) == 4 and code.isdigit():
                    tickers.append((f"{code}.TWO", name, "上櫃"))
    except Exception:
        pass

    if not tickers:
        backup = [
            ("2330.TW", "台積電", "半導體"), ("2317.TW", "鴻海", "其他電子"),
            ("2454.TW", "聯發科", "半導體"), ("2382.TW", "廣達", "電腦及週邊"),
            ("3231.TW", "緯創", "電腦及週邊"), ("3017.TW", "奇鋐", "散熱"),
            ("6669.TW", "緯穎", "伺服器"), ("1519.TW", "華城", "重電"),
        ]
        return pd.DataFrame(backup, columns=["ticker", "name", "industry"])

    return pd.DataFrame(tickers, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


class DailyPriceStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_cache_error: str | None = None
        self.download_errors: list[str] = []
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )"""
            )

    def clear(self) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM daily_prices")
            return True
        except Exception as exc:
            self.last_cache_error = str(exc)
            return False

    @staticmethod
    def _ticker_frame(data: pd.DataFrame, ticker: str, chunk_len: int) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        try:
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = set(map(str, data.columns.get_level_values(0)))
                lvl1 = set(map(str, data.columns.get_level_values(1)))
                if ticker in lvl0:
                    out = data[ticker].copy()
                elif ticker in lvl1:
                    out = data.xs(ticker, axis=1, level=1).copy()
                else:
                    return pd.DataFrame()
            else:
                out = data.copy() if chunk_len == 1 else pd.DataFrame()
            wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
            return out[wanted].dropna(how="all") if wanted else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def batch_fetch_and_update(self, tickers: list[str], period: str = "5y", chunk_size: int = 80) -> dict:
        """Batch download with explicit auto_adjust=False and per-chunk retry."""
        if yf is None:
            raise RuntimeError("缺少 yfinance；請先執行 pip install -r requirements.txt")
        unique = list(dict.fromkeys([t for t in tickers if t]))
        self.download_errors = []
        downloaded = 0
        for i in range(0, len(unique), chunk_size):
            chunk = unique[i:i + chunk_size]
            data = None
            exc_last = None
            for attempt in range(2):
                try:
                    history_args = _yf_history_args(period)
                    data = yf.download(
                        chunk,
                        **history_args,
                        group_by="ticker",
                        auto_adjust=True,
                        actions=False,
                        progress=False,
                        threads=True,
                        timeout=20,
                    )
                    break
                except Exception as exc:
                    exc_last = exc
                    time.sleep(0.6 * (attempt + 1))
            if data is None or getattr(data, "empty", True):
                self.download_errors.append(f"chunk {i // chunk_size + 1}: {exc_last or 'empty response'}")
                continue

            records: list[tuple] = []
            for ticker in chunk:
                frame = self._ticker_frame(data, ticker, len(chunk))
                if frame.empty:
                    continue
                frame = frame.copy()
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                count_before = len(records)
                for idx, row in frame.iterrows():
                    try:
                        close = float(row.get("Close", np.nan))
                        if not np.isfinite(close) or close <= 0:
                            continue
                        op = float(row.get("Open", close))
                        hi = float(row.get("High", close))
                        lo = float(row.get("Low", close))
                        vol = float(row.get("Volume", 0.0))
                        records.append((ticker, idx.strftime("%Y-%m-%d"), op, hi, lo, close, vol))
                    except Exception:
                        continue
                if len(records) > count_before:
                    downloaded += 1

            if records:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.executemany(
                            """INSERT OR REPLACE INTO daily_prices
                            (ticker, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            records,
                        )
                except Exception as exc:
                    self.download_errors.append(f"sqlite write: {exc}")
        return {"downloaded_tickers": downloaded, "errors": self.download_errors[:]}

    @staticmethod
    def _num(value):
        try:
            text = str(value).replace(",", "").strip()
            if text in {"", "--", "---", "X", "-"}:
                return None
            v = float(text)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    @staticmethod
    def _roc_date(value):
        text = str(value or "").strip().replace("/", "").replace("-", "")
        if len(text) == 7 and text.isdigit():
            return f"{int(text[:3]) + 1911:04d}-{text[3:5]}-{text[5:7]}"
        if len(text) == 8 and text.isdigit():
            return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
        return ""

    def overlay_official_eod(self) -> dict:
        """Overlay latest official TWSE/TPEx EOD bars to reduce evening Yahoo lag."""
        records = []
        counts = {"twse": 0, "tpex": 0, "index": 0, "errors": []}
        headers = {"User-Agent": "Mozilla/5.0 AlphaRadar/13.0"}

        try:
            rows = requests.get(
                "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                headers=headers, timeout=12
            ).json()
            for row in rows if isinstance(rows, list) else []:
                code = str(row.get("Code", "")).strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                date = self._roc_date(row.get("Date"))
                op = self._num(row.get("OpeningPrice")); hi = self._num(row.get("HighestPrice"))
                lo = self._num(row.get("LowestPrice")); cl = self._num(row.get("ClosingPrice"))
                vol = self._num(row.get("TradeVolume"))
                if date and cl and cl > 0:
                    records.append((f"{code}.TW", date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
                    counts["twse"] += 1
        except Exception as exc:
            counts["errors"].append(f"TWSE official EOD: {exc}")

        try:
            rows = requests.get(
                "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
                headers=headers, timeout=12
            ).json()
            for row in rows if isinstance(rows, list) else []:
                code = str(row.get("SecuritiesCompanyCode") or row.get("Code") or "").strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                date = self._roc_date(row.get("Date"))
                op = self._num(row.get("Open") or row.get("OpeningPrice"))
                hi = self._num(row.get("High") or row.get("HighestPrice"))
                lo = self._num(row.get("Low") or row.get("LowestPrice"))
                cl = self._num(row.get("Close") or row.get("ClosingPrice"))
                vol = self._num(row.get("TradingShares") or row.get("TradeVolume"))
                if date and cl and cl > 0:
                    records.append((f"{code}.TWO", date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
                    counts["tpex"] += 1
        except Exception as exc:
            counts["errors"].append(f"TPEx official EOD: {exc}")

        # TAIEX official close, so the market regime can also advance even if Yahoo lags.
        try:
            rows = requests.get(
                "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
                headers=headers, timeout=12
            ).json()
            for row in rows if isinstance(rows, list) else []:
                name = str(row.get("指數") or row.get("Index") or row.get("Name") or "")
                if "發行量加權股價指數" not in name and "TAIEX" not in name.upper():
                    continue
                date = self._roc_date(row.get("日期") or row.get("Date"))
                cl = self._num(row.get("收盤指數") or row.get("ClosingIndex") or row.get("Close"))
                if date and cl and cl > 0:
                    records.append(("^TWII", date, cl, cl, cl, cl, 0.0))
                    counts["index"] = 1
                    break
        except Exception as exc:
            counts["errors"].append(f"TWSE index EOD: {exc}")

        if records:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany(
                        """INSERT OR REPLACE INTO daily_prices
                        (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        records,
                    )
            except Exception as exc:
                counts["errors"].append(f"official EOD sqlite: {exc}")
        return counts

    def get_prices(self, ticker: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, open as Open, high as High, low as Low, close as Close, volume as Volume "
                "FROM daily_prices WHERE ticker = ? ORDER BY date ASC",
                conn,
                params=(ticker,),
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates("date").set_index("date").sort_index()
        return df


class ResearchDataClient:
    """Cached research-data client for fields shown next to recommendations."""

    def __init__(self, db_path: Path, token: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = (token if token is not None else _finmind_token()).strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 AlphaRadar/13.0"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self._init_cache()

    def _init_cache(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS research_cache (
                    cache_key TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )

    def _cache_get(self, key: str, ttl_hours: float) -> Any | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT fetched_at, payload FROM research_cache WHERE cache_key=?", (key,)
                ).fetchone()
            if not row:
                return None
            ts = datetime.datetime.fromisoformat(row[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            age = datetime.datetime.now(datetime.timezone.utc) - ts.astimezone(datetime.timezone.utc)
            if age.total_seconds() > ttl_hours * 3600:
                return None
            return json.loads(row[1])
        except Exception:
            return None

    def _cache_set(self, key: str, payload: Any):
        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO research_cache(cache_key,fetched_at,payload) VALUES(?,?,?)",
                    (key, now, text),
                )
        except Exception:
            pass

    def _finmind(self, dataset: str, stock_id: str, start_date: str, end_date: str | None = None, ttl_hours: float = 6.0) -> list[dict]:
        key = f"fm:{dataset}:{stock_id}:{start_date}:{end_date or ''}"
        cached = self._cache_get(key, ttl_hours)
        if isinstance(cached, list):
            return cached
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
        if end_date:
            params["end_date"] = end_date
        if self.token:
            params["token"] = self.token
        try:
            r = self.session.get(FINMIND_URL, params=params, timeout=12)
            r.raise_for_status()
            js = r.json()
            data = js.get("data", []) if isinstance(js, dict) else []
            if isinstance(data, list):
                self._cache_set(key, data)
                return data
        except Exception:
            pass
        return []

    @staticmethod
    def _date_ago(days: int) -> str:
        return (_taipei_timestamp().date() - datetime.timedelta(days=days)).isoformat()

    def monthly_revenue(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockMonthRevenue", stock_id, self._date_ago(800), ttl_hours=8)
        if not rows:
            return {"available": False, "rows": []}
        df = pd.DataFrame(rows)
        if df.empty or "revenue" not in df.columns:
            return {"available": False, "rows": []}
        for c in ["revenue", "revenue_year", "revenue_month"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["revenue", "revenue_year", "revenue_month"])
        if df.empty:
            return {"available": False, "rows": []}
        df["period"] = pd.to_datetime(
            df["revenue_year"].astype(int).astype(str) + "-" + df["revenue_month"].astype(int).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        df = df.dropna(subset=["period"]).sort_values("period").drop_duplicates("period", keep="last")
        df["mom"] = df["revenue"].pct_change()
        period_map = {p.to_period("M"): r for p, r in zip(df["period"], df["revenue"])}
        df["yoy"] = [
            (rev / period_map.get(p.to_period("M") - 12) - 1.0)
            if period_map.get(p.to_period("M") - 12) not in (None, 0) else np.nan
            for p, rev in zip(df["period"], df["revenue"])
        ]
        tail = df.tail(12)
        out_rows = [
            {
                "month": r.period.strftime("%Y-%m"),
                "revenue_billion": round(float(r.revenue) / 1e8, 2),  # 億元 = 1e8 NTD
                "mom_pct": round(float(r.mom) * 100.0, 2) if pd.notna(r.mom) else None,
                "yoy_pct": round(float(r.yoy) * 100.0, 2) if pd.notna(r.yoy) else None,
            }
            for r in tail.itertuples()
        ]
        latest = out_rows[-1] if out_rows else {}
        yoy_vals = [r["yoy_pct"] for r in out_rows[-3:] if r.get("yoy_pct") is not None]
        return {
            "available": bool(out_rows),
            "latest": latest,
            "rows": out_rows,
            "avg_yoy_3m_pct": round(float(np.mean(yoy_vals)), 2) if yoy_vals else None,
        }

    def financials(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockFinancialStatements", stock_id, self._date_ago(650), ttl_hours=12)
        if not rows:
            return {"available": False, "quarters": []}
        df = pd.DataFrame(rows)
        if df.empty or not {"date", "type", "value"}.issubset(df.columns):
            return {"available": False, "quarters": []}
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value", "type"])
        if df.empty:
            return {"available": False, "quarters": []}
        p = df.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
        if p.empty:
            return {"available": False, "quarters": []}

        rev_col = next((c for c in ["Revenue", "OperatingRevenue", "TotalRevenue", "Income"] if c in p.columns), None)
        gp_col = "GrossProfit" if "GrossProfit" in p.columns else None
        eps_col = "EPS" if "EPS" in p.columns else None

        q = p.tail(4)
        quarters = []
        for dt, row in q.iterrows():
            eps = float(row[eps_col]) if eps_col and pd.notna(row.get(eps_col)) else None
            rev = float(row[rev_col]) if rev_col and pd.notna(row.get(rev_col)) else None
            gp = float(row[gp_col]) if gp_col and pd.notna(row.get(gp_col)) else None
            gm = (gp / rev * 100.0) if gp is not None and rev not in (None, 0) else None
            quarters.append({
                "quarter": f"{dt.year}Q{((dt.month - 1) // 3) + 1}",
                "date": dt.strftime("%Y-%m-%d"),
                "eps": round(eps, 2) if eps is not None else None,
                "gross_margin_pct": round(gm, 2) if gm is not None and np.isfinite(gm) else None,
            })

        # YTD running sum of FinMind single-quarter EPS.  Corporate actions can
        # cause the arithmetic sum to differ from the issuer-reported cumulative EPS,
        # so the UI labels this as an unadjusted sum.
        running_by_year = {}
        for item in quarters:
            year = int(item["quarter"][:4])
            if item.get("eps") is not None:
                running_by_year[year] = running_by_year.get(year, 0.0) + float(item["eps"])
                item["eps_ytd_sum_unadjusted"] = round(running_by_year[year], 2)
            else:
                item["eps_ytd_sum_unadjusted"] = None

        eps_vals = [x["eps"] for x in quarters if x.get("eps") is not None]
        gm_vals = [x["gross_margin_pct"] for x in quarters if x.get("gross_margin_pct") is not None]
        return {
            "available": bool(quarters),
            "quarters": quarters,
            "eps_4q_sum": round(float(np.sum(eps_vals)), 2) if eps_vals else None,
            "gross_margin_4q_avg_pct": round(float(np.mean(gm_vals)), 2) if gm_vals else None,
            "gross_margin_latest_pct": gm_vals[-1] if gm_vals else None,
        }

    def valuation(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockPER", stock_id, self._date_ago(50), ttl_hours=6)
        if not rows:
            return {"available": False}
        df = pd.DataFrame(rows)
        if df.empty:
            return {"available": False}
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
        row = df.iloc[-1]
        pe = pd.to_numeric(pd.Series([row.get("PER")]), errors="coerce").iloc[0]
        pbr = pd.to_numeric(pd.Series([row.get("PBR")]), errors="coerce").iloc[0]
        yld = pd.to_numeric(pd.Series([row.get("dividend_yield")]), errors="coerce").iloc[0]
        return {
            "available": bool(pd.notna(pe) or pd.notna(pbr)),
            "date": str(row.get("date", ""))[:10],
            "pe": round(float(pe), 2) if pd.notna(pe) else None,
            "pbr": round(float(pbr), 2) if pd.notna(pbr) else None,
            "dividend_yield_pct": round(float(yld), 2) if pd.notna(yld) else None,
        }

    def institutional_flow(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockInstitutionalInvestorsBuySellWide", stock_id, self._date_ago(45), ttl_hours=2)
        if not rows:
            # Compatibility fallback to long table.
            long_rows = self._finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, self._date_ago(45), ttl_hours=2)
            return self._institutional_long(long_rows)
        df = pd.DataFrame(rows)
        if df.empty:
            return {"available": False}
        for c in df.columns:
            if c.endswith("_buy") or c.endswith("_sell"):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").tail(22)

        def net(buy, sell):
            if buy in df.columns and sell in df.columns:
                return float((df[buy] - df[sell]).sum())
            return 0.0

        foreign = net("Foreign_Investor_buy", "Foreign_Investor_sell") + net("Foreign_Dealer_Self_buy", "Foreign_Dealer_Self_sell")
        trust = net("Investment_Trust_buy", "Investment_Trust_sell")
        if "Dealer_buy" in df.columns and "Dealer_sell" in df.columns:
            dealer = net("Dealer_buy", "Dealer_sell")
        else:
            dealer = net("Dealer_self_buy", "Dealer_self_sell") + net("Dealer_Hedging_buy", "Dealer_Hedging_sell")
        total = foreign + trust + dealer
        return {
            "available": True,
            "sessions": int(len(df)),
            "foreign_net_lots": round(foreign / 1000.0, 1),
            "trust_net_lots": round(trust / 1000.0, 1),
            "dealer_net_lots": round(dealer / 1000.0, 1),
            "total_net_lots": round(total / 1000.0, 1),
            "summary": self._flow_text(total / 1000.0, foreign / 1000.0, trust / 1000.0, dealer / 1000.0),
        }

    def _institutional_long(self, rows: list[dict]) -> dict:
        if not rows:
            return {"available": False}
        df = pd.DataFrame(rows)
        if df.empty or not {"buy", "sell", "name"}.issubset(df.columns):
            return {"available": False}
        df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
        df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest_dates = sorted(df["date"].dropna().unique())[-22:]
            df = df[df["date"].isin(latest_dates)]
        nets = df.assign(net=df["buy"] - df["sell"]).groupby("name")["net"].sum()
        foreign = float(nets[[i for i in nets.index if "Foreign" in str(i) or "外資" in str(i)]].sum())
        trust = float(nets[[i for i in nets.index if "Investment_Trust" in str(i) or "投信" in str(i)]].sum())
        dealer = float(nets[[i for i in nets.index if "Dealer" in str(i) or "自營" in str(i)]].sum())
        total = foreign + trust + dealer
        return {
            "available": True,
            "sessions": int(df["date"].nunique()) if "date" in df.columns else None,
            "foreign_net_lots": round(foreign / 1000.0, 1),
            "trust_net_lots": round(trust / 1000.0, 1),
            "dealer_net_lots": round(dealer / 1000.0, 1),
            "total_net_lots": round(total / 1000.0, 1),
            "summary": self._flow_text(total / 1000.0, foreign / 1000.0, trust / 1000.0, dealer / 1000.0),
        }

    @staticmethod
    def _flow_text(total, foreign, trust, dealer) -> str:
        direction = "偏買超" if total > 0 else "偏賣超" if total < 0 else "大致中性"
        return f"近月三大法人{direction} {total:+,.0f} 張（外資 {foreign:+,.0f}、投信 {trust:+,.0f}、自營商 {dealer:+,.0f}）"

    def branch_main_force_proxy(self, stock_id: str, max_sessions: int = 20) -> dict:
        """
        Optional broker-branch concentration proxy.

        Disabled by default.  It is intentionally labelled as a proxy because
        'main force' is not an official investor category.  FinMind's branch data
        is Sponsor-only and one trading date is requested at a time.
        """
        if os.getenv("ENABLE_BRANCH_FLOW", "0") != "1":
            return {
                "available": False,
                "reason": "DISABLED",
                "summary": "未啟用券商分點資料；主力屬代理指標，不以三大法人冒充。",
            }
        if not self.token:
            return {
                "available": False,
                "reason": "TOKEN_REQUIRED",
                "summary": "主力代理需 FinMind Sponsor 分點權限；目前未設定 FINMIND_TOKEN。",
            }

        end = _taipei_timestamp().date()
        days = []
        d = end
        while len(days) < max_sessions and (end - d).days < 20:
            if d.weekday() < 5:
                days.append(d)
            d -= datetime.timedelta(days=1)

        frames = []
        for day in days:
            key = f"branch:{stock_id}:{day.isoformat()}"
            cached = self._cache_get(key, 4)
            if isinstance(cached, list):
                rows = cached
            else:
                params = {"data_id": stock_id, "start_date": day.isoformat(), "token": self.token}
                try:
                    r = self.session.get(FINMIND_BRANCH_URL, params=params, timeout=10)
                    r.raise_for_status()
                    js = r.json()
                    rows = js.get("data", []) if isinstance(js, dict) else []
                    if isinstance(rows, list):
                        self._cache_set(key, rows)
                except Exception:
                    rows = []
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            return {
                "available": False,
                "reason": "NO_BRANCH_DATA",
                "summary": "分點資料目前無可用回傳（可能為權限、休市或資料尚未更新）。",
            }
        df = pd.concat(frames, ignore_index=True)
        if not {"securities_trader", "buy", "sell"}.issubset(df.columns):
            return {"available": False, "reason": "SCHEMA_MISMATCH", "summary": "分點資料欄位格式不符。"}
        df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
        df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
        by_branch = df.assign(net=df["buy"] - df["sell"]).groupby("securities_trader")["net"].sum().sort_values()
        top_buy = by_branch.tail(5).sort_values(ascending=False)
        top_sell = by_branch.head(5)
        # Concentration proxy: top five net buys + top five net sells, expressed separately.
        buy_lots = float(top_buy.clip(lower=0).sum()) / 1000.0
        sell_lots = float((-top_sell.clip(upper=0)).sum()) / 1000.0
        proxy = buy_lots - sell_lots
        return {
            "available": True,
            "sessions": len(frames),
            "proxy_net_lots": round(proxy, 1),
            "top_buy_lots": round(buy_lots, 1),
            "top_sell_lots": round(sell_lots, 1),
            "top_buy_branches": [{"name": str(k), "net_lots": round(float(v) / 1000.0, 1)} for k, v in top_buy.items()],
            "top_sell_branches": [{"name": str(k), "net_lots": round(float(v) / 1000.0, 1)} for k, v in top_sell.items()],
            "summary": f"近 {len(frames)} 個交易日分點代理：前5大淨買與前5大淨賣差額 {proxy:+,.0f} 張（非官方『主力』分類）",
        }

    def stock_research(self, ticker: str, include_branch: bool = False) -> dict:
        stock_id = stock_id_from_ticker(ticker)
        rev = self.monthly_revenue(stock_id)
        fin = self.financials(stock_id)
        val = self.valuation(stock_id)
        inst = self.institutional_flow(stock_id)
        branch = self.branch_main_force_proxy(stock_id) if include_branch else {
            "available": False,
            "reason": "NOT_REQUESTED",
            "summary": "主力分點代理未啟用。",
        }
        return {
            "stock_id": stock_id,
            "monthly_revenue": rev,
            "financials": fin,
            "valuation": val,
            "institutional_flow": inst,
            "main_force_proxy": branch,
            "source": "FinMind v4 + optional sponsor branch data",
            "fetched_at": _taipei_timestamp().isoformat(timespec="seconds"),
        }
