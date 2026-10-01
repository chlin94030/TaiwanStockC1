"""
Taiwan Alpha Radar V13.1 - plain-language Streamlit research dashboard.
Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import gc
import html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp, provider_runtime_status
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed
from presentation import investment_view, plain_summary
from industry_profile import profile_note
from policy_engine import HORIZONS
from return_first_model import holding_review

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))

VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股診斷"]
HORIZON_LABELS = {"short": "短線｜約2週", "mid": "中線｜約2個月", "long": "長波段｜約半年"}
FAMILY_LABELS = {
    "完整模式｜價格＋公司表現＋法人": "full",
    "價格模式｜只看走勢與歷史案例": "price_only",
    "公司模式｜再看營收／EPS／毛利率": "business_confirmed",
    "資金模式｜再看近月三大法人": "flow_confirmed",
}
SETUP_LABELS = {
    "BREAKOUT": "平台突破",
    "PULLBACK": "趨勢回測",
    "RECLAIM": "重新站回",
    "TREND": "趨勢延續",
    "BASE": "整理觀察",
    "DRYUP": "低量沉澱",
}
STATE_LABELS = {
    "CONDITIONS_MET_NOT_FILLED": "條件已接近／符合 · 仍以次一交易日價格確認",
    "WAIT_ENTRY_ZONE": "等待回到布局區",
    "WAIT_BREAKOUT": "等待突破確認",
    "WAIT_CONFIRMATION": "等待量價確認",
    "DO_NOT_CHASE": "超出追價上限 · 等回測",
    "INVALIDATED": "結構失效",
    "DATA_UNVERIFIED": "資料待核對",
    "NO_RETURN_ESTIMATE": "報酬樣本不足",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格資料未核對，暫不作留／賣判斷。",
    "ORIGINAL_STRUCTURE_INVALIDATED": "⚠️ 已跌破原始防守價，原交易結構失效。",
    "ORIGINAL_THESIS_INVALIDATED": "⚠️ 原買進理由已失效，應重新檢視部位。",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "🔔 已觸及移動保護價，請依原定紀律檢視執行。",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始停損或保護價，需手動評估。",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "✅ 尚未破壞原設定的防守結構；不代表未來報酬保證。",
}
REGIME_LABELS = {"BULL": "🟢 大盤偏多", "NEUTRAL": "🟡 大盤震盪", "BEAR": "🔴 大盤偏弱", "UNKNOWN": "⚪ 大盤待確認"}

CSS = """
<style>
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --soft:#f8fafc; --blue:#2563eb; }
.stApp { background:#f8fafc; color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.block-container { max-width:1180px; padding-top:1.25rem; padding-bottom:5rem; }
.hero { padding:30px 28px; border-radius:26px; background:linear-gradient(135deg,#0f172a,#1e293b 58%,#1d4ed8); color:white; margin-bottom:20px; box-shadow:0 18px 35px rgba(15,23,42,.15); }
.hero .eyebrow { letter-spacing:.14em; font-size:.82rem; font-weight:800; color:#bfdbfe; }
.hero h1 { font-size:2.2rem; font-weight:900; margin:.45rem 0; color:white; letter-spacing:-.025em; }
.hero p { margin:0; color:#e2e8f0; line-height:1.6; }
.statusline { display:inline-block; background:white; border:1px solid var(--line); border-radius:12px; padding:9px 14px; margin:4px 0 16px; color:#475569; font-weight:700; }
.card { background:white; border:1px solid var(--line); border-radius:22px; padding:24px; margin:18px 0 8px; box-shadow:0 8px 24px rgba(15,23,42,.045); }
.card-head { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
.card-title { font-size:1.65rem; font-weight:900; letter-spacing:-.02em; }
.card-code { color:var(--muted); margin-top:4px; font-size:.95rem; font-weight:600; }
.card-price { font-size:1.55rem; font-weight:900; text-align:right; white-space:nowrap; }
.badge { display:inline-block; font-size:.88rem; font-weight:800; padding:6px 11px; border-radius:10px; background:#f1f5f9; color:#475569; margin:10px 6px 6px 0; }
.badge-blue { background:#dbeafe; color:#1d4ed8; }
.badge-green { background:#d1fae5; color:#047857; }
.badge-amber { background:#fef3c7; color:#92400e; }
.topnote { margin:13px 0; padding:13px 16px; border-radius:14px; background:#eff6ff; color:#1e40af; font-weight:650; line-height:1.55; }
.verdict { margin:14px 0; padding:16px 18px; border-radius:16px; background:#f8fafc; border:1px solid #e2e8f0; }
.verdict-title { font-size:1.2rem; font-weight:900; color:#0f172a; margin-bottom:5px; }
.verdict-summary { color:#334155; font-weight:650; line-height:1.55; }
.reason-list { margin:10px 0 0 0; padding-left:1.2rem; color:#334155; line-height:1.6; }
.risk-note { margin-top:10px; padding:9px 11px; border-radius:10px; background:#fff7ed; color:#9a3412; font-size:.9rem; line-height:1.5; }
.industry-tag { display:inline-block; font-size:.9rem; font-weight:850; padding:7px 12px; border-radius:999px; background:#eef2ff; color:#4338ca; margin:8px 6px 4px 0; }
.return-box { margin:12px 0; padding:17px 20px; border-radius:17px; background:linear-gradient(135deg,#f0f9ff,#e0f2fe); border:1px solid #bae6fd; }
.return-k { color:#0369a1; font-weight:800; }
.return-v { font-size:2.65rem; line-height:1.1; font-weight:900; color:#0284c7; margin:4px 0; }
.return-desc { color:#0369a1; font-weight:650; }
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:12px 0; }
.stat { background:#fafafa; border:1px solid #f1f5f9; border-radius:14px; padding:12px 14px; }
.stat .k { color:var(--muted); font-size:.82rem; font-weight:700; line-height:1.35; }
.stat .v { color:var(--ink); font-size:1.18rem; font-weight:900; margin-top:5px; }
.research-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:14px 0 4px; }
.research { background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px 14px; }
.research .k { color:#64748b; font-size:.82rem; font-weight:750; }
.research .v { color:#0f172a; font-size:1.06rem; font-weight:900; margin-top:5px; line-height:1.35; }
.research .s { color:#64748b; font-size:.78rem; margin-top:3px; line-height:1.35; }
.decision { padding:14px 16px; border-radius:14px; background:#fef3c7; border:1px solid #fde68a; color:#92400e; font-weight:850; margin-top:12px; }
.decision-ok { background:#d1fae5; border-color:#a7f3d0; color:#065f46; }
.levels { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0; }
.level { background:#f8fafc; border:1px solid var(--line); border-radius:12px; padding:12px; }
.level .k { color:#64748b; font-size:.82rem; font-weight:700; }
.level .v { color:#0f172a; font-size:1.08rem; font-weight:900; margin-top:4px; }
.small-note { color:#64748b; font-size:.86rem; line-height:1.5; }
@media(max-width:760px){
 .block-container{padding:1rem .8rem 4rem}.hero{padding:22px 18px}.hero h1{font-size:1.7rem}.card{padding:18px 14px}
 .card-title{font-size:1.35rem}.card-price{font-size:1.25rem}.stats{grid-template-columns:repeat(2,1fr)}
 .research-grid{grid-template-columns:repeat(2,1fr)}.levels{grid-template-columns:repeat(2,1fr)}.return-v{font-size:2.2rem}
}
</style>
"""


def esc(v):
    return html.escape(str(v if v is not None else ""))


def finite(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def pct_decimal(v, signed=True):
    x = finite(v)
    if x is None:
        return "—"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"


def pct_point(v, signed=True):
    x = finite(v)
    if x is None:
        return "—"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def num(v, digits=2):
    x = finite(v)
    return "—" if x is None else f"{x:,.{digits}f}".rstrip("0").rstrip(".")


def signed_lots(v):
    x = finite(v)
    return "—" if x is None else f"{x:+,.0f} 張"


def _latest_research(obj):
    r = obj.get("research") or {}
    rev = r.get("monthly_revenue") or {}
    fin = r.get("financials") or {}
    val = r.get("valuation") or {}
    inst = r.get("institutional_flow") or {}
    main = r.get("main_force_proxy") or {}
    return r, rev, fin, val, inst, main


def render_chart(chart, plan, key):
    if not chart or "ohlcv" not in chart:
        st.caption("這檔股票的 K 線尚未在本機快取中。")
        return
    df = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart.get("dates", [])
    if df.empty:
        return

    close, high, low = df["Close"], df["High"], df["Low"]
    for n in [5, 20, 60, 120]:
        df[f"MA{n}"] = close.rolling(n).mean()
    std20 = close.rolling(20).std()
    df["BBU"] = df["MA20"] + 2 * std20
    df["BBL"] = df["MA20"] - 2 * std20
    low9, high9 = low.rolling(9).min(), high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["HIST"] = df["MACD"] - df["SIGNAL"]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.46, 0.13, 0.13, 0.14, 0.14],
        subplot_titles=("K線 / MA / Bollinger", "成交量", "KD", "MACD", "RSI"),
    )
    fig.add_trace(go.Candlestick(
        x=dates, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
        decreasing_line_color="#10b981", decreasing_fillcolor="#10b981", name="K線"
    ), row=1, col=1)
    colors = {5:"#f59e0b",20:"#2563eb",60:"#9333ea",120:"#ec4899"}
    for n in [5,20,60,120]:
        fig.add_trace(go.Scatter(x=dates, y=df[f"MA{n}"], line=dict(color=colors[n], width=1.25), name=f"MA{n}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.BBU, line=dict(color="#94a3b8", width=1, dash="dot"), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.BBL, line=dict(color="#94a3b8", width=1, dash="dot"), name="布林下軌"), row=1, col=1)
    if plan:
        fig.add_hrect(y0=plan.get("zone_low"), y1=plan.get("zone_high"), fillcolor="rgba(37,99,235,.10)", line_width=0, row=1, col=1)
        fig.add_hline(y=plan.get("trigger"), line_color="#d97706", line_dash="dot", row=1, col=1)
        fig.add_hline(y=plan.get("invalidation"), line_color="#e11d48", line_dash="dash", row=1, col=1)
    vol_colors = np.where(df.Close >= df.Open, "#ef4444", "#10b981")
    fig.add_trace(go.Bar(x=dates, y=df.Volume, marker_color=vol_colors, name="量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.K, line=dict(color="#2563eb", width=1.3), name="K"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.D, line=dict(color="#f59e0b", width=1.3), name="D"), row=3, col=1)
    fig.add_hline(y=80, line_color="#cbd5e1", line_dash="dash", row=3, col=1)
    fig.add_hline(y=20, line_color="#cbd5e1", line_dash="dash", row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MACD, line=dict(color="#2563eb", width=1.2), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.SIGNAL, line=dict(color="#f59e0b", width=1.2), name="Signal"), row=4, col=1)
    hist_colors = np.where(df.HIST >= 0, "#ef4444", "#10b981")
    fig.add_trace(go.Bar(x=dates, y=df.HIST, marker_color=hist_colors, name="Hist"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.RSI, line=dict(color="#059669", width=1.3), name="RSI"), row=5, col=1)
    fig.add_hline(y=70, line_color="#cbd5e1", line_dash="dash", row=5, col=1)
    fig.add_hline(y=30, line_color="#cbd5e1", line_dash="dash", row=5, col=1)
    fig.update_layout(height=800, margin=dict(l=8,r=8,t=26,b=8), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                      xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type="category", nticks=7, fixedrange=True, showgrid=False)
    fig.update_yaxes(fixedrange=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar":False,"displaylogo":False})


def _research_html(obj):
    _, rev, fin, val, inst, main = _latest_research(obj)
    latest_rev = rev.get("latest") or {}
    rev_v = "—"
    rev_s = "近一年月營收未取得"
    if latest_rev:
        rev_v = f"{num(latest_rev.get('revenue_billion'))} 億"
        rev_s = f"{latest_rev.get('month','')}｜MoM {pct_point(latest_rev.get('mom_pct'))}｜YoY {pct_point(latest_rev.get('yoy_pct'))}"
    eps_v = f"{num(fin.get('eps_4q_sum'))} 元" if fin.get("eps_4q_sum") is not None else "—"
    eps_s = "近4季單季EPS加總（未校正配股/分割）"
    gm_v = pct_point(fin.get("gross_margin_latest_pct"), signed=False) if fin.get("gross_margin_latest_pct") is not None else "—"
    gm_s = f"近4季平均 {pct_point(fin.get('gross_margin_4q_avg_pct'), signed=False)}" if fin.get("gross_margin_4q_avg_pct") is not None else "毛利率未取得／不適用"
    pe_v = num(val.get("pe")) + " 倍" if val.get("pe") is not None else "—"
    pe_s = f"資料日 {val.get('date','—')}"
    inst_v = signed_lots(inst.get("total_net_lots")) if inst.get("available") else "—"
    inst_s = "近月三大法人合計" if inst.get("available") else "近月法人資料未取得"
    main_v = signed_lots(main.get("proxy_net_lots")) if main.get("available") else "未啟用"
    main_s = (f"近{main.get('sessions',20)}日券商分點代理" if main.get("available") else "需額外券商分點資料權限")
    return f"""
    <div class="research-grid">
      <div class="research"><div class="k">最新月營收</div><div class="v">{esc(rev_v)}</div><div class="s">{esc(rev_s)}</div></div>
      <div class="research"><div class="k">近4季 EPS 合計</div><div class="v">{esc(eps_v)}</div><div class="s">{esc(eps_s)}</div></div>
      <div class="research"><div class="k">毛利率</div><div class="v">{esc(gm_v)}</div><div class="s">{esc(gm_s)}</div></div>
      <div class="research"><div class="k">目前本益比</div><div class="v">{esc(pe_v)}</div><div class="s">{esc(pe_s)}</div></div>
      <div class="research"><div class="k">三大法人近1月</div><div class="v">{esc(inst_v)}</div><div class="s">{esc(inst_s)}</div></div>
      <div class="research"><div class="k">主力近1月（代理）</div><div class="v">{esc(main_v)}</div><div class="s">{esc(main_s)}</div></div>
    </div>"""


def _render_research_details(obj, key):
    _, rev, fin, val, inst, main = _latest_research(obj)
    with st.expander("🧾 展開公司數據與資金明細", expanded=False):
        if rev.get("rows"):
            st.markdown("**近 12 個月營收**")
            rdf = pd.DataFrame(rev["rows"]).rename(columns={
                "month":"月份", "revenue_billion":"營收(億元)", "mom_pct":"MoM(%)", "yoy_pct":"YoY(%)"
            })
            st.dataframe(rdf, hide_index=True, use_container_width=True)
        else:
            st.caption("未取得近一年月營收資料。")

        if fin.get("quarters"):
            st.markdown("**近一年各季 EPS／累積 EPS／毛利率**")
            qdf = pd.DataFrame(fin["quarters"]).rename(columns={
                "quarter":"季別", "eps":"單季EPS", "eps_ytd_sum_unadjusted":"當年度累積EPS*", "gross_margin_pct":"毛利率(%)"
            })
            cols = [c for c in ["季別", "單季EPS", "當年度累積EPS*", "毛利率(%)"] if c in qdf.columns]
            st.dataframe(qdf[cols], hide_index=True, use_container_width=True)
            st.caption("* FinMind 的 EPS 為單季值；此處累積為單季直接加總。若期間有配股或股票分割，可能與公司財報列示之累積 EPS 不一致。")
        else:
            st.caption("未取得近一年季財報資料，或該業別毛利率不適用。")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**近一月三大法人**")
            if inst.get("available"):
                idf = pd.DataFrame([
                    ["外資", inst.get("foreign_net_lots")],
                    ["投信", inst.get("trust_net_lots")],
                    ["自營商", inst.get("dealer_net_lots")],
                    ["合計", inst.get("total_net_lots")],
                ], columns=["類別", "買賣超(張)"])
                st.dataframe(idf, hide_index=True, use_container_width=True)
                st.caption(inst.get("summary", ""))
            else:
                st.caption("未取得三大法人資料。")
        with c2:
            st.markdown("**近一月主力買賣超代理**")
            if main.get("available"):
                st.metric("券商分點代理淨額", signed_lots(main.get("proxy_net_lots")))
                st.caption(main.get("summary", ""))
                if main.get("top_buy_branches"):
                    bdf = pd.DataFrame(main.get("top_buy_branches") + main.get("top_sell_branches", []))
                    st.dataframe(bdf.rename(columns={"name":"券商分點", "net_lots":"淨買賣超(張)"}), hide_index=True, use_container_width=True)
            else:
                st.caption(main.get("summary", "主力代理未啟用。"))
                st.caption("主力沒有交易所統一正式分類；本系統只在取得券商分點資料時做近似觀察，不會用三大法人代替主力。")

        st.markdown("**進階｜模型分數拆解**")
        # Find the first horizon with the current card key isn't available here; show all three compactly.
        score_rows = []
        for h in HORIZONS:
            b = obj.get("horizons", {}).get(h, {})
            comp = b.get("score_components") or {}
            score_rows.append({
                "週期": HORIZON_LABELS[h],
                "總排名分": b.get("ranking_score"),
                "技術": comp.get("technical"),
                "歷史相似情境": comp.get("empirical"),
                "基本面": comp.get("fundamental"),
                "法人": comp.get("flow"),
            })
        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)


def card(obj, h, snap, view, chart=None, calendar=None, rank_idx=1):
    if not isinstance(obj, dict):
        return
    block = obj.get("horizons", {}).get(h, {})
    f = block.get("forecast") or {}
    plan = block.get("plan")
    s = f.get("strategy") or {}
    condition = block.get("entry_state", "NO_RETURN_ESTIMATE")
    calendar = calendar or calendar_reference(DATA_DIR, now=_taipei_timestamp())
    fresh = daily_freshness(obj.get("price_date", ""), calendar)
    review_ok = entry_review_allowed(obj.get("price_date", ""), plan, calendar)

    empirical_med = s.get("median") if f.get("estimate_available") else None
    pos_rate = s.get("historical_positive_rate") if f.get("estimate_available") else None
    sample_n = int(f.get("local_effective_n") or 0)
    fine = obj.get("fine_industry") or obj.get("industry") or "產業定位待補"
    view_info = investment_view(obj, h, snap.get("market", {}) if isinstance(snap, dict) else {})

    reason_items = "".join(f"<li>{esc(x)}</li>" for x in view_info.get("reasons", []))
    caution_items = " ".join(esc(x) for x in view_info.get("cautions", []))
    risk_html = f'<div class="risk-note"><b>要注意：</b>{caution_items}</div>' if caution_items else ""

    action_text = {
        "CONDITIONS_MET_NOT_FILLED": "現在怎麼做：列入優先觀察，下一個交易日守穩再決定。",
        "WAIT_ENTRY_ZONE": "現在怎麼做：先等拉回，不必追價。",
        "WAIT_BREAKOUT": "現在怎麼做：先觀察，等股價真正轉強再考慮。",
        "WAIT_CONFIRMATION": "現在怎麼做：先等價格與買盤一起轉強。",
        "DO_NOT_CHASE": "現在怎麼做：不要追高，等價格回到比較合理的位置。",
        "INVALIDATED": "現在怎麼做：先不進場，等走勢重新站穩。",
        "DATA_UNVERIFIED": "現在怎麼做：先更新資料，再做判斷。",
        "NO_RETURN_ESTIMATE": "現在怎麼做：歷史案例不足，先觀察。",
    }.get(condition, "現在怎麼做：先觀察，不急著進場。")
    if not fresh.get("current_daily"):
        action_text = f"資料不是最新交易日（目前 {obj.get('price_date','—')}），請先更新後再判斷。"

    horizon_result = {
        "short": "約2週後",
        "mid": "約2個月後",
        "long": "約半年後",
    }.get(h, "之後")

    st.markdown(f"""
    <div class="card">
      <div class="card-head">
        <div><div class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker','')))}</div>
        <div class="card-code">{esc(obj.get('ticker',''))} · {esc(obj.get('industry',''))}</div></div>
        <div class="card-price">{num(obj.get('price'))} 元<div class="card-code">{esc(obj.get('price_date',''))} 收盤</div></div>
      </div>
      <span class="industry-tag">{esc(fine)}</span>
      <span class="badge badge-blue">{esc(HORIZON_LABELS.get(h,h))}</span>

      <div class="verdict">
        <div class="verdict-title">投資總評：{esc(view_info.get('title','觀察'))}</div>
        <div class="verdict-summary">{esc(view_info.get('summary',''))}</div>
        <ul class="reason-list">{reason_items}</ul>
        {risk_html}
      </div>

      <div class="return-box">
        <div class="return-k">過去類似走勢，{esc(horizon_result)}的中間結果</div>
        <div class="return-v">{pct_decimal(empirical_med) if empirical_med is not None else '案例不足'}</div>
        <div class="return-desc">這不是目標價；只是把過去最相近的走勢拿來看，實際結果可能更好或更差。</div>
      </div>
      <div class="stats">
        <div class="stat"><div class="k">近20日比大盤</div><div class="v">{pct_decimal(f.get('alpha_mean'))}</div></div>
        <div class="stat"><div class="k">類似情況上漲比例</div><div class="v">{pct_decimal(pos_rate, signed=False) if pos_rate is not None else '—'}</div></div>
        <div class="stat"><div class="k">可比歷史案例</div><div class="v">{sample_n:,} 次</div></div>
        <div class="stat"><div class="k">較差情況平均跌幅</div><div class="v">{pct_decimal(s.get('expected_shortfall10_loss')) if s else '—'}</div></div>
      </div>
      {_research_html(obj)}
      <div class="decision {'decision-ok' if review_ok and condition=='CONDITIONS_MET_NOT_FILLED' else ''}">{esc(action_text)}</div>
    </div>
    """, unsafe_allow_html=True)

    _render_research_details(obj, f"research_{view}_{h}_{obj.get('ticker')}")

    with st.expander("📊 進階｜K線與買賣價位", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
              <div class="level"><div class="k">較理想的觀察區</div><div class="v">{num(plan.get('zone_low'))}–{num(plan.get('zone_high'))}</div></div>
              <div class="level"><div class="k">轉強確認價</div><div class="v">{num(plan.get('trigger'))}</div></div>
              <div class="level"><div class="k">不建議追高超過</div><div class="v">{num(plan.get('chase_limit'))}</div></div>
              <div class="level"><div class="k">跌破需重新檢視</div><div class="v">{num(plan.get('invalidation'))}</div></div>
            </div>""", unsafe_allow_html=True)
        snap_id = snap.get("snapshot_id", "default") if isinstance(snap, dict) else "default"
        chart = chart or (snap.get("charts", {}).get(obj.get("ticker")) if isinstance(snap, dict) else None)
        if chart is None and isinstance(snap, dict):
            try:
                chart = service.chart_on_demand(snap, obj.get("ticker", ""), DATA_DIR, allow_fetch=False)
            except Exception:
                chart = None
        render_chart(chart, plan, f"chart_{view}_{h}_{obj.get('ticker')}_{snap_id}")

def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS[h])
    if not snap or not isinstance(snap, dict):
        st.info("尚無研究快照，請先更新市場資料。")
        return
    picked = service.select_market_best(snap, h, n=5)
    if not picked:
        st.info("目前沒有可排序標的。")
        return
    st.caption("顯示目前相對值得先看的 5 檔；不是每一檔都適合立刻買，卡片會直接告訴你要觀察、等拉回，還是不要追高。")
    for idx, obj in enumerate(picked, 1):
        card(obj, h, snap, h, calendar=calendar, rank_idx=idx)


def main():
    st.set_page_config(page_title="Alpha Radar V13.1", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div class="eyebrow">TAIWAN ALPHA RADAR · V13.1</div>
      <h1>先告訴你「值不值得看」，再看數字</h1>
      <p>全台股掃描 · 短中長線各選 5 檔 · 細產業定位 · 投資總評與白話推薦理由 · 營收 / EPS / 毛利率 / 本益比 / 法人資金</p>
    </div>""", unsafe_allow_html=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if st.session_state.get("alpha_version") != service.OPERATIONS_VERSION:
        for key in ("alpha_snapshot", "alpha_doctor", "alpha_error"):
            st.session_state.pop(key, None)
        st.session_state["alpha_version"] = service.OPERATIONS_VERSION
        previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json", include_features=False)
        if previous and previous.get("model_version") == service.OPERATIONS_VERSION:
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(previous)

    with st.sidebar:
        st.markdown("### 掃描設定")
        family_label = st.selectbox("想納入哪些資訊", list(FAMILY_LABELS), index=0, key="family_v13")
        period = st.selectbox("拿多少年歷史來比較", ["5y", "8y", "10y", "3y"], index=0, key="period_v13")
        candidate_size = st.selectbox("深入比較股票數", [400, 700, 1000], index=1, key="candidate_v13")
        research_pool = st.selectbox("補充公司資料的候選數", [5, 10, 15], index=1, key="research_pool_v13")
        status = provider_runtime_status()
        st.caption(f"FinMind Token：{'已設定' if status['finmind_token_configured'] else '未設定（免費資料仍會嘗試）'}")
        st.caption(f"券商分點主力代理：{'已啟用' if status['branch_flow_enabled'] else '未啟用'}")
        st.caption("如要主力代理：設定 FINMIND_TOKEN（Sponsor 權限）及 ENABLE_BRANCH_FLOW=1。")
        with st.expander("快取維護", expanded=False):
            if st.button("清除行情與舊快照", key="clear_v13"):
                DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
                service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
                st.session_state.pop("alpha_snapshot", None)
                st.success("已清除行情與舊快照。")

    settings = service.RunSettings(
        reference_size=600,
        candidate_size=int(candidate_size),
        research_pool_per_horizon=int(research_pool),
        history_period=period,
        model_family=FAMILY_LABELS[family_label],
    )

    if st.button("⚡ 更新市場並重新選股", type="primary", use_container_width=True, key="scan_v13"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1.0, max(0.0, float(value))), text=stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(snap)
            st.session_state.pop("alpha_doctor", None)
            st.session_state.pop("alpha_error", None)
            del snap
            gc.collect()
        except Exception as exc:
            st.session_state["alpha_error"] = f"{type(exc).__name__}: {exc}"
            st.error(f"掃描未完成，上一份成功快照仍保留。錯誤：{type(exc).__name__}: {exc}")
        finally:
            progress.empty()

    snap = st.session_state.get("alpha_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())
    if snap and isinstance(snap, dict):
        m = snap.get("market", {})
        cov = snap.get("coverage", {})
        st.markdown(
            f"<div class='statusline'>資料日 <b>{esc(snap.get('price_date','—'))}</b> · {REGIME_LABELS.get(m.get('regime'),'⚪ 大盤待確認')} · 掃描 {cov.get('requested',0):,} 檔 · 可交易條件通過 {cov.get('liquid',0):,} 檔 · 深入比較 {snap.get('candidate_n',0):,} 檔</div>",
            unsafe_allow_html=True,
        )
        if cov.get("errors"):
            st.caption(f"行情下載有 {len(cov['errors'])} 個批次警告；排序僅使用成功取得且通過流動性檢查的標的。")
    err = st.session_state.get("alpha_error")
    if err:
        st.warning(err)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v13")
    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 一眼看懂：短、中、長線代表標的")
        if not snap:
            st.info("尚無研究快照。")
        else:
            used = set()
            for h in HORIZONS:
                picks = service.select_market_best(snap, h, n=5)
                obj = next((x for x in picks if x.get("ticker") not in used), picks[0] if picks else None)
                if obj:
                    used.add(obj.get("ticker"))
                    rank = next((i for i, x in enumerate(picks, 1) if x.get("ticker") == obj.get("ticker")), 1)
                    card(obj, h, snap, "prime", calendar=calendar, rank_idx=rank)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 個股診斷與持股紀律")
        with st.form("doctor_v13"):
            code = st.text_input("股票代碼", value="2330", help="如 2330 或 6187")
            horizon_text = st.selectbox("評估週期", list(HORIZON_LABELS.values()), index=1)
            own = st.checkbox("已有持股")
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("買進成本（元/股）", min_value=0.0, value=0.0)
                invalid = st.number_input("原始停損價（元/股）", min_value=0.0, value=0.0)
            with c2:
                shares = st.number_input("持股股數", min_value=0, value=0, step=1000)
                trail = st.number_input("移動獲利保護價（元/股）", min_value=0.0, value=0.0)
            thesis = st.selectbox("原買進理由是否仍成立？", ["尚未確認", "看多理由仍成立", "看多理由已不成立"])
            submitted = st.form_submit_button("立即診斷", type="primary", use_container_width=True)
        if submitted:
            dr = service.diagnose(code, snap, DATA_DIR)
            dr["h"] = next(k for k, v in HORIZON_LABELS.items() if v == horizon_text)
            st.session_state["alpha_doctor"] = service.compact_doctor_result(dr)
        dr = st.session_state.get("alpha_doctor")
        if dr:
            if dr.get("error"):
                st.error(f"診斷資料不足：{dr.get('error')}")
            elif dr.get("stock"):
                card(dr["stock"], dr.get("h", "mid"), snap or {}, "doctor", calendar=calendar)
                if own:
                    result = holding_review(
                        dr["stock"]["price"],
                        original_invalidation=invalid if invalid > 0 else None,
                        trailing_protection=trail if trail > 0 else None,
                        thesis_broken=(thesis == "看多理由已不成立"),
                    )
                    pnl = None
                    if cost > 0:
                        pnl = (dr["stock"]["price"] / cost - 1.0) * 100.0
                    st.markdown("#### 持股紀律檢查")
                    st.info(HOLD_LABELS.get(result, result) + (f"｜目前損益約 {pnl:+.2f}%" if pnl is not None else ""))
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

    st.caption(profile_note() + "｜歷史案例只用來幫助比較，不代表未來一定會重演；買賣仍需自行判斷價格與風險。")


if __name__ == "__main__":
    main()
