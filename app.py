"""
app.py
QuantEdge AI — AI-Powered Portfolio Analyst
Bloomberg-dark-themed Streamlit app backed by Claude + yfinance.

Incorporates Cougar Global Investments methodology:
- Probability of Loss as lead risk metric (PMPT)
- Three-scenario macro framework: Bull / Base / Bear
- 20 Global Asset Class grid: In / Out / Watch
- Tactical rebalancing signals driven by macro regime
- Downside-first framing throughout
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()


def _get_env_api_key() -> str:
    """
    Resolve the Anthropic API key from multiple sources (priority order):
    1. Streamlit Cloud secrets  (st.secrets["ANTHROPIC_API_KEY"])
    2. Local .env / shell env   (os.environ["ANTHROPIC_API_KEY"])
    Safe to call before the Streamlit app has fully initialised.
    """
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")

from portfolio_engine import (
    DEFAULT_ROTH_HOLDINGS,
    DEFAULT_INVESTABLE_CASH,
    WATCHLIST_RECOMMENDATIONS,
    MACRO_SCENARIOS,
    ASSET_CLASS_UNIVERSE,
    fetch_prices_batch,
    build_portfolio,
    portfolio_summary,
    compute_risk_metrics,
    monte_carlo_simulation,
    fetch_macro_data,
    fetch_portfolio_movers,
    generate_allocation_plan,
    generate_scenario_allocation_plan,
    rebalancing_alerts,
    assess_macro_regime,
    get_asset_class_grid,
    get_tactical_rebalancing_signals,
)
from analyst import QuantEdgeAnalyst

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="QuantEdge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS — Bloomberg Dark Theme
# ─────────────────────────────────────────────

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0e1a !important;
    color: #e0e6f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1220 !important;
    border-right: 1px solid #1e2d4a !important;
}
h1 { color: #00d4ff !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { color: #e0e6f0 !important; font-weight: 600 !important; }
h3 { color: #a0b4cc !important; font-weight: 500 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1a2e 0%, #111827 100%) !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #6b8099 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

[data-testid="stTabs"] button { color: #6b8099 !important; font-weight: 500 !important; border-bottom: 2px solid transparent !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; background: transparent !important; }

.stButton > button {
    background: linear-gradient(135deg, #0066aa 0%, #003d7a 100%) !important;
    color: #ffffff !important; border: 1px solid #00d4ff !important;
    border-radius: 6px !important; font-weight: 600 !important;
    letter-spacing: 0.02em !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff 0%, #0066aa 100%) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background-color: #0d1a2e !important; border: 1px solid #1e2d4a !important;
    color: #e0e6f0 !important; border-radius: 6px !important;
}
[data-testid="stDataFrame"] { border: 1px solid #1e2d4a !important; border-radius: 8px !important; }
.streamlit-expanderHeader {
    background-color: #0d1a2e !important; border: 1px solid #1e2d4a !important;
    border-radius: 8px !important; color: #a0b4cc !important;
}
.stAlert { border-radius: 8px !important; border-left: 4px solid #00d4ff !important; background-color: #0d1a2e !important; }
.status-online  { color: #00ff88; font-weight: 600; }
.status-offline { color: #ff4d4d; font-weight: 600; }
.analyst-response {
    background: linear-gradient(135deg, #0a1525 0%, #0d1a2e 100%);
    border: 1px solid #1e4a6b; border-left: 4px solid #00d4ff;
    border-radius: 10px; padding: 24px;
    font-size: 0.95rem; line-height: 1.7;
}
hr { border-color: #1e2d4a !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }

/* Regime badges */
.regime-bull  { background:#0a2a1a; border:1px solid #00ff88; color:#00ff88; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; display:inline-block; }
.regime-base  { background:#0a1a2a; border:1px solid #00d4ff; color:#00d4ff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; display:inline-block; }
.regime-bear  { background:#2a0a0a; border:1px solid #ff4d4d; color:#ff4d4d; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; display:inline-block; }

/* Asset class status badges */
.ac-in    { background:#0a2a1a; border:1px solid #00ff88; color:#00ff88; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; }
.ac-watch { background:#1a1a0a; border:1px solid #ffd700; color:#ffd700; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; }
.ac-out   { background:#1a0a0a; border:1px solid #4a2a2a; color:#6b3a3a; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; }

/* POL prominence */
.pol-display {
    background: linear-gradient(135deg, #1a0a0a 0%, #0d1220 100%);
    border: 2px solid #ff4d4d; border-radius: 12px;
    padding: 20px 24px; text-align: center;
}
.pol-value { font-size: 2.8rem; font-weight: 800; color: #ff4d4d; font-family: 'JetBrains Mono', monospace; }
.pol-label { color: #6b8099; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
.pol-low   { color: #00ff88 !important; border-color: #00ff88 !important; }
.pol-mid   { color: #ffd700 !important; border-color: #ffd700 !important; }

/* Signal rows */
.signal-add      { border-left: 4px solid #00ff88; }
.signal-exit     { border-left: 4px solid #ff4d4d; }
.signal-increase { border-left: 4px solid #00d4ff; }
.signal-reduce   { border-left: 4px solid #ffd700; }
.signal-hold     { border-left: 4px solid #4a5a6a; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Plotly Theme
# ─────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1220",
    font=dict(family="Inter, sans-serif", color="#a0b4cc", size=12),
    title_font=dict(color="#e0e6f0", size=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d4a", font=dict(color="#a0b4cc")),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#111827", showline=False, tickfont=dict(color="#6b8099")),
    yaxis=dict(gridcolor="#111827", showline=False, tickfont=dict(color="#6b8099")),
)

COLORS = {
    "blue":   "#00d4ff", "gold":   "#ffd700", "green":  "#00ff88",
    "red":    "#ff4d4d", "purple": "#9d4edd", "orange": "#ff7c2a",
    "teal":   "#00b4a0", "gray":   "#6b8099",
}
PALETTE = [COLORS["blue"], COLORS["gold"], COLORS["green"], COLORS["purple"],
           COLORS["orange"], COLORS["teal"], "#ff69b4", "#adb5bd"]


# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────

for key, val in {
    "portfolio_df":    None,
    "risk_metrics":    None,
    "macro_data":      None,
    "analysis_result": None,
    "regime_result":   None,
    "chat_history":    [],
    "last_refresh":    None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def fmt_currency(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.2f}"

def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:+.{decimals}f}%"

def pol_color(pct):
    """Color code probability of loss."""
    try:
        p = float(pct)
        if p <= 10:  return COLORS["green"]
        if p <= 20:  return COLORS["gold"]
        return COLORS["red"]
    except:
        return COLORS["gray"]

def regime_badge(regime: str) -> str:
    sc = MACRO_SCENARIOS.get(regime, {})
    css = f"regime-{regime}"
    icon = sc.get("icon", "")
    label = sc.get("label", regime.upper())
    return f'<span class="{css}">{icon} {label}</span>'

@st.cache_data(ttl=300)
def load_market_data(tickers: tuple):
    return fetch_prices_batch(list(tickers))

@st.cache_data(ttl=600)
def load_macro():
    return fetch_macro_data()


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px;">
        <div style="font-size:2.2rem;">⚡</div>
        <div style="font-size:1.3rem; font-weight:700; color:#00d4ff; letter-spacing:0.05em;">QUANTEDGE AI</div>
        <div style="font-size:0.65rem; color:#6b8099; letter-spacing:0.15em; margin-top:2px;">TACTICAL ALLOCATION · PMPT</div>
    </div>
    <hr style="border-color:#1e2d4a; margin-bottom:20px;">
    """, unsafe_allow_html=True)

    st.markdown("**🔑 Anthropic API Key**")
    api_key_input = st.text_input(
        "API Key", type="password",
        value=_get_env_api_key(),
        placeholder="sk-ant-...", label_visibility="collapsed",
        help="Loaded from Streamlit secrets or local .env automatically. Paste here to override.",
    )

    analyst = QuantEdgeAnalyst(api_key=api_key_input if api_key_input else None)
    status_icon = "🟢" if analyst.available else "🔴"
    status_class = "status-online" if analyst.available else "status-offline"
    st.markdown(
        f'<div style="font-size:0.8rem; margin-bottom:16px;">'
        f'{status_icon} Analyst: <span class="{status_class}">{analyst.status}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**📊 Portfolio Settings**")

    investable_cash = st.number_input(
        "Investable Cash ($)", min_value=0.0, max_value=100000.0,
        value=DEFAULT_INVESTABLE_CASH, step=100.0, format="%.0f",
    )
    roth_base = st.number_input(
        "Roth IRA Value ($)", min_value=0.0, max_value=100000.0,
        value=200.0, step=10.0, format="%.0f",
    )
    annual_contribution = st.number_input(
        "Annual Contribution ($)", min_value=0.0, max_value=7000.0,
        value=500.0, step=100.0,
    )

    st.markdown("---")
    st.markdown("**🎯 Macro Regime Override**")
    regime_override = st.selectbox(
        "Force regime (or leave Auto)",
        ["Auto-detect", "bull", "base", "bear"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        for k in ["portfolio_df","risk_metrics","macro_data","analysis_result","regime_result"]:
            st.session_state[k] = None
        st.rerun()

    if st.session_state.last_refresh:
        st.markdown(
            f'<div style="font-size:0.72rem; color:#6b8099; text-align:center;">'
            f'Last refresh: {st.session_state.last_refresh}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem; color:#6b8099; text-align:center;">'
        'QuantEdge AI v2.0 · Cougar Global Methodology<br>For educational purposes only.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Load Data
# ─────────────────────────────────────────────

all_tickers = tuple(h["ticker"] for h in DEFAULT_ROTH_HOLDINGS)

with st.spinner("Fetching live market data..."):
    if st.session_state.portfolio_df is None:
        prices = load_market_data(all_tickers)
        st.session_state.portfolio_df = build_portfolio(DEFAULT_ROTH_HOLDINGS, prices)
        st.session_state.last_refresh = time.strftime("%H:%M:%S")

    if st.session_state.macro_data is None:
        st.session_state.macro_data = load_macro()

portfolio_df = st.session_state.portfolio_df
macro_data   = st.session_state.macro_data
summary      = portfolio_summary(portfolio_df)
alerts       = rebalancing_alerts(portfolio_df)

# Macro regime
algo_regime = assess_macro_regime(macro_data)
if regime_override != "Auto-detect":
    active_regime = regime_override
    algo_regime["regime"] = regime_override
    algo_regime["confidence"] = 100
else:
    active_regime = algo_regime["regime"]

current_scenario    = MACRO_SCENARIOS[active_regime]
allocation_plan     = generate_scenario_allocation_plan(investable_cash, active_regime)
rebalancing_signals = get_tactical_rebalancing_signals(portfolio_df, active_regime, investable_cash)
asset_class_grid    = get_asset_class_grid(active_regime)


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.markdown("""
    <h1 style="margin-bottom:0;">⚡ QuantEdge AI</h1>
    <p style="color:#6b8099; font-size:0.88rem; margin-top:4px;">
    Tactical Asset Allocation · Post Modern Portfolio Theory · Cougar Global Methodology
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.metric("Total Portfolio", fmt_currency(summary["total_value"] + investable_cash))
with col_h3:
    st.markdown(
        f'<div style="padding:8px 0; text-align:center;">'
        f'<div style="color:#6b8099; font-size:0.72rem; text-transform:uppercase; margin-bottom:6px;">Macro Regime</div>'
        f'{regime_badge(active_regime)}'
        f'<div style="color:#6b8099; font-size:0.72rem; margin-top:4px;">{algo_regime["confidence"]}% confidence</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🎯 Macro Framework",
    "🤖 AI Analyst",
    "📈 Roadmap",
    "🌍 Market Intel",
    "💬 Chat",
])


# ═══════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD  (downside-first layout)
# ═══════════════════════════════════════════════════════

with tab1:

    # ── PMPT Lead Metrics ──
    st.markdown("#### 🛡️ Downside Risk — Primary Metrics (PMPT)")
    st.markdown(
        '<div style="color:#6b8099; font-size:0.82rem; margin-bottom:12px;">'
        'Per Post Modern Portfolio Theory (PMPT) — Cougar Global\'s framework — '
        'Probability of Loss is shown first. Standard deviation penalizes upside; only downside risk matters.'
        '</div>', unsafe_allow_html=True,
    )

    # Load risk metrics
    if st.session_state.risk_metrics is None:
        with st.spinner("Computing PMPT risk metrics..."):
            st.session_state.risk_metrics = compute_risk_metrics(portfolio_df)
    risk_metrics = st.session_state.risk_metrics

    pol_1yr = risk_metrics.get("prob_of_loss_1yr")
    pol_3yr = risk_metrics.get("prob_of_loss_3yr")
    dd_ann  = risk_metrics.get("downside_deviation")
    sortino = risk_metrics.get("sortino")
    mdd     = risk_metrics.get("max_drawdown")

    pol1_color = pol_color(pol_1yr) if pol_1yr else COLORS["gray"]
    pol3_color = pol_color(pol_3yr) if pol_3yr else COLORS["gray"]

    pm1, pm2, pm3, pm4, pm5 = st.columns(5)

    with pm1:
        pol_val = f"{pol_1yr:.1f}%" if isinstance(pol_1yr, float) else "N/A"
        low_cls = "pol-low" if isinstance(pol_1yr, float) and pol_1yr <= 10 else (
                  "pol-mid" if isinstance(pol_1yr, float) and pol_1yr <= 20 else "")
        st.markdown(
            f'<div class="pol-display {low_cls}">'
            f'<div class="pol-value" style="color:{pol1_color};">{pol_val}</div>'
            f'<div class="pol-label">Prob. of Loss (1yr)</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with pm2:
        pol3_val = f"{pol_3yr:.1f}%" if isinstance(pol_3yr, float) else "N/A"
        st.markdown(
            f'<div class="pol-display">'
            f'<div class="pol-value" style="color:{pol3_color};">{pol3_val}</div>'
            f'<div class="pol-label">Prob. of Loss (3yr)</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with pm3:
        st.metric("Downside Deviation", f"{dd_ann}%" if isinstance(dd_ann, float) else "N/A",
                  help="Annualized deviation of negative returns only (PMPT). Lower = better.")
    with pm4:
        st.metric("Sortino Ratio", str(sortino) if sortino != "N/A" else "N/A",
                  help="Excess return per unit of downside deviation. Cougar Global's preferred risk-adjusted metric.")
    with pm5:
        st.metric("Max Drawdown", f"{mdd}%" if isinstance(mdd, float) else "N/A",
                  help="Worst peak-to-trough loss. The key bear market survival metric.")

    st.markdown("")
    st.markdown("#### 📐 Full Risk Dashboard")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    secondary = [
        (r1, "Sharpe Ratio",    risk_metrics.get("sharpe"),           ""),
        (r2, "Beta (vs SPY)",   risk_metrics.get("beta"),             ""),
        (r3, "Ann. Volatility", risk_metrics.get("volatility"),       "%"),
        (r4, "Ann. Return",     risk_metrics.get("annualized_return"), "%"),
        (r5, "# Positions",     summary["num_positions"],             ""),
        (r6, "ETF Weight",      f"{summary['etf_weight']:.1f}",       "%"),
    ]
    for col, label, val, sfx in secondary:
        with col:
            display = f"{val}{sfx}" if isinstance(val, (int, float)) else str(val)
            st.metric(label, display)

    # ── Alerts ──
    if alerts:
        st.markdown("")
        for alert in alerts:
            if alert["type"] == "danger":   st.error(f"⚠️ {alert['msg']}")
            elif alert["type"] == "warning": st.warning(f"📢 {alert['msg']}")
            else:                            st.info(f"ℹ️ {alert['msg']}")

    st.markdown("---")

    # ── Bear Floor / Bull Ceiling ──
    initial_value = (summary["total_value"] or roth_base) + investable_cash
    mc_quick = monte_carlo_simulation(initial_value, years=5, simulations=300)

    floor_col, med_col, ceil_col, pol_mc_col = st.columns(4)
    with floor_col:
        st.metric("🐻 Bear Floor (5yr, 5th %ile)",
                  fmt_currency(mc_quick["bear_floor"]),
                  delta=fmt_pct((mc_quick["bear_floor"]/initial_value-1)*100) if initial_value else None,
                  delta_color="normal")
    with med_col:
        st.metric("⚖️ Base Case (5yr median)",
                  fmt_currency(mc_quick["final_p50"]),
                  delta=fmt_pct((mc_quick["final_p50"]/initial_value-1)*100) if initial_value else None)
    with ceil_col:
        st.metric("🐂 Bull Ceiling (5yr, 95th %ile)",
                  fmt_currency(mc_quick["bull_ceiling"]),
                  delta=fmt_pct((mc_quick["bull_ceiling"]/initial_value-1)*100) if initial_value else None)
    with pol_mc_col:
        pol_mc = mc_quick["prob_of_loss_by_year"][5]
        st.markdown(
            f'<div class="pol-display" style="margin-top:8px;">'
            f'<div class="pol-value" style="font-size:2rem; color:{pol_color(pol_mc)};">{pol_mc:.1f}%</div>'
            f'<div class="pol-label">MC Prob. of Loss (5yr)</div>'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Charts ──
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("#### 🥧 Portfolio Allocation")
        valid_df = portfolio_df.dropna(subset=["Value"])
        if not valid_df.empty:
            fig = go.Figure(go.Pie(
                labels=valid_df["Ticker"], values=valid_df["Value"],
                textinfo="label+percent", hole=0.45,
                marker=dict(colors=PALETTE[:len(valid_df)], line=dict(color="#0a0e1a", width=2)),
                textfont=dict(color="#e0e6f0", size=13),
            ))
            fig.add_annotation(text=f"${valid_df['Value'].sum():,.0f}",
                               x=0.5, y=0.5, font=dict(size=18, color="#00d4ff"), showarrow=False)
            fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False, title="Roth IRA by Value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Live prices unavailable — check internet connection.")

    with chart2:
        st.markdown("#### 🏭 Sector Breakdown")
        if not valid_df.empty:
            sec_df = valid_df.groupby("Sector")["Value"].sum().reset_index().sort_values("Value")
            fig = go.Figure(go.Bar(
                x=sec_df["Value"], y=sec_df["Sector"], orientation="h",
                marker=dict(color=sec_df["Value"], colorscale=[[0,"#0d1a2e"],[1,"#00d4ff"]],
                            line=dict(color="#00d4ff", width=0.5)),
                text=[fmt_currency(v) for v in sec_df["Value"]],
                textposition="outside", textfont=dict(color="#a0b4cc", size=11),
            ))
            fig.update_layout(**PLOT_LAYOUT, height=320, title="Allocation by Sector")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Holdings Table ──
    st.markdown("#### 📋 Holdings Detail")
    disp = portfolio_df[["Ticker","Name","Shares","Price","Value","Change%","Weight%","Sector","Type"]].copy()
    disp["Price"]   = disp["Price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    disp["Value"]   = disp["Value"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    disp["Change%"] = disp["Change%"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    disp["Weight%"] = disp["Weight%"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown(f"#### 💰 ${investable_cash:,.0f} Deployment — {current_scenario['icon']} {active_regime.title()} Regime")

    for item in allocation_plan:
        priority_color = COLORS["blue"] if item["rank"] <= 2 else COLORS["gold"] if item["rank"] <= 4 else COLORS["gray"]
        st.markdown(
            f"""<div style="background:#0d1a2e; border:1px solid #1e2d4a; border-left:3px solid {priority_color};
            border-radius:8px; padding:14px 18px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-family:'JetBrains Mono',monospace; color:#00d4ff; font-weight:700; font-size:1rem;">
                    #{item['rank']} {item['ticker']}</span>
                    <span style="color:#6b8099; font-size:0.85rem; margin-left:8px;">{item['name']}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#ffd700; font-weight:700; font-size:1.1rem;">${item['amount']:,.0f}</span>
                    <span style="color:#6b8099; font-size:0.8rem; margin-left:6px;">{item['alloc_pct']}%</span>
                    <span style="margin-left:10px; padding:2px 8px; border-radius:4px; font-size:0.75rem;
                    background:#0a0e1a; border:1px solid {priority_color}; color:{priority_color};">
                    {item['priority']}</span>
                </div>
            </div>
            <div style="color:#a0b4cc; font-size:0.83rem; margin-top:6px;">{item['rationale']}</div>
            </div>""", unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
#  TAB 2 — MACRO FRAMEWORK  (NEW — Cougar Global core)
# ═══════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🎯 Cougar Global Macro Framework")
    st.markdown(
        '<div style="color:#6b8099; font-size:0.88rem; margin-bottom:16px;">'
        'Three-scenario tactical allocation system. The macro regime determines which of the 20 global asset classes '
        'are "in" or "out" of the model portfolio. Participate in bull markets, avoid bear markets.'
        '</div>', unsafe_allow_html=True,
    )

    # ── Regime Header ──
    r_conf = algo_regime["confidence"]
    r_signals = algo_regime.get("signal_log", [])
    sc = current_scenario

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0d1a2e,#0a1525); border:1px solid #1e4a6b; '
        f'border-radius:12px; padding:20px 24px; margin-bottom:20px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
        f'<div>'
        f'<div style="color:#6b8099; font-size:0.75rem; text-transform:uppercase; margin-bottom:8px;">Active Macro Regime</div>'
        f'{regime_badge(active_regime)}'
        f'<span style="color:#6b8099; font-size:0.8rem; margin-left:12px;">{r_conf}% confidence</span>'
        f'<div style="color:#a0b4cc; font-size:0.88rem; margin-top:12px; max-width:600px;">{sc["description"]}</div>'
        f'</div>'
        f'<div style="text-align:right; min-width:180px;">'
        f'<div style="color:#6b8099; font-size:0.72rem; text-transform:uppercase; margin-bottom:8px;">Active Asset Classes</div>'
        f'<div style="color:#00d4ff; font-size:2rem; font-weight:800;">{len(sc["active_asset_ids"])}</div>'
        f'<div style="color:#6b8099; font-size:0.75rem;">of 20 in model</div>'
        f'</div>'
        f'</div>'
        f'</div>', unsafe_allow_html=True,
    )

    # Regime signals
    if r_signals:
        st.markdown("**Regime Signals Detected:**")
        for sig in r_signals:
            icon = "🟢" if "bullish" in sig.lower() else ("🔴" if "bearish" in sig.lower() else "🟡")
            st.markdown(f"&nbsp;&nbsp;{icon} {sig}")

    # ── AI Regime Assessment ──
    if st.button("🧠 AI Macro Regime Assessment", type="primary"):
        with st.spinner("Running Cougar Global 3-step macro process..."):
            regime_commentary = analyst.macro_regime_assessment(macro_data, algo_regime)
            st.session_state.regime_result = regime_commentary

    if st.session_state.regime_result:
        st.markdown(
            f'<div class="analyst-response">{st.session_state.regime_result}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Three Scenarios Side by Side ──
    st.markdown("#### 📋 All Three Scenario Allocations")
    sc_cols = st.columns(3)
    sc_keys = ["bull", "base", "bear"]
    sc_colors = [COLORS["green"], COLORS["blue"], COLORS["red"]]

    for col, sk, sc_color in zip(sc_cols, sc_keys, sc_colors):
        sc_data = MACRO_SCENARIOS[sk]
        is_active = (sk == active_regime)
        border = f"2px solid {sc_color}" if is_active else f"1px solid #1e2d4a"
        glow   = f"box-shadow: 0 0 20px {sc_color}33;" if is_active else ""
        with col:
            active_label = " ← ACTIVE" if is_active else ""
            st.markdown(
                f'<div style="background:#0d1a2e; border:{border}; border-radius:10px; '
                f'padding:16px; {glow}">'
                f'<div style="color:{sc_color}; font-weight:700; font-size:1rem; margin-bottom:4px;">'
                f'{sc_data["icon"]} {sc_data["label"]}{active_label}</div>'
                f'<div style="color:#6b8099; font-size:0.75rem; margin-bottom:12px;">'
                f'{len(sc_data["active_asset_ids"])} asset classes active</div>',
                unsafe_allow_html=True,
            )
            for alloc in sc_data["allocations"]:
                bar_w = alloc["weight"] * 2  # scale to px
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.8rem;">'
                    f'<span style="color:#e0e6f0; font-family:\'JetBrains Mono\',monospace;">{alloc["etf"]}</span>'
                    f'<span style="color:{sc_color}; font-weight:600;">{alloc["weight"]}%</span></div>'
                    f'<div style="background:#0a0e1a; border-radius:3px; height:4px; margin-top:2px;">'
                    f'<div style="background:{sc_color}; width:{bar_w}px; height:4px; border-radius:3px; max-width:100%;"></div>'
                    f'</div></div>', unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tactical Rebalancing Signals ──
    st.markdown("#### 🔔 Tactical Rebalancing Signals")
    st.markdown(
        f'<div style="color:#6b8099; font-size:0.82rem; margin-bottom:12px;">'
        f'Comparing current holdings to {active_regime.upper()} regime target allocations. '
        f'Cougar Global rotates positions when macro signals shift regime.'
        f'</div>', unsafe_allow_html=True,
    )

    action_colors = {"ADD": COLORS["green"], "EXIT": COLORS["red"], "INCREASE": COLORS["blue"],
                     "REDUCE": COLORS["gold"], "HOLD": COLORS["gray"]}
    action_icons  = {"ADD": "➕", "EXIT": "🚪", "INCREASE": "📈", "REDUCE": "✂️", "HOLD": "✅"}

    if rebalancing_signals:
        for sig in rebalancing_signals:
            ac = action_colors.get(sig["action"], COLORS["gray"])
            ai = action_icons.get(sig["action"], "•")
            css_cls = f"signal-{sig['action'].lower()}"
            urgency_color = COLORS["red"] if sig["urgency"] == "High" else (COLORS["gold"] if sig["urgency"] == "Medium" else COLORS["gray"])
            st.markdown(
                f'<div style="background:#0d1a2e; border:1px solid #1e2d4a; border-left:4px solid {ac}; '
                f'border-radius:8px; padding:12px 16px; margin-bottom:8px; '
                f'display:flex; justify-content:space-between; align-items:center;">'
                f'<div>'
                f'<span style="color:{ac}; font-weight:700; font-size:0.85rem;">{ai} {sig["action"]}</span>'
                f'&nbsp;&nbsp;<span style="color:#00d4ff; font-family:\'JetBrains Mono\',monospace; font-weight:700;">{sig["ticker"]}</span>'
                f'&nbsp;<span style="color:#6b8099; font-size:0.8rem;">{sig["label"]}</span>'
                f'<div style="color:#a0b4cc; font-size:0.82rem; margin-top:4px;">{sig["msg"]}</div>'
                f'</div>'
                f'<div style="text-align:right; min-width:100px;">'
                f'<span style="color:{urgency_color}; font-size:0.75rem; padding:2px 8px; '
                f'border:1px solid {urgency_color}; border-radius:4px;">{sig["urgency"]}</span>'
                f'<div style="color:#6b8099; font-size:0.75rem; margin-top:4px;">'
                f'{sig["current"]:.1f}% → {sig["target"]}%</div>'
                f'</div>'
                f'</div>', unsafe_allow_html=True,
            )
    else:
        st.info("No rebalancing signals — portfolio aligned with current regime.")

    st.markdown("---")

    # ── 20 Asset Class Grid ──
    st.markdown("#### 🌐 20 Global Asset Class Universe")
    st.markdown(
        f'<div style="color:#6b8099; font-size:0.82rem; margin-bottom:14px;">'
        f'Cougar Global\'s universe of 20 asset classes — typically fewer than 10 are active at once. '
        f'Status reflects current {active_regime.upper()} regime.'
        f'</div>', unsafe_allow_html=True,
    )

    # Group by category
    categories = ["US Equity", "Intl Equity", "Real Assets", "Fixed Income", "Defensive"]
    for cat in categories:
        cat_assets = [a for a in asset_class_grid if a["category"] == cat]
        if not cat_assets:
            continue
        st.markdown(f"**{cat}**")
        cols = st.columns(min(len(cat_assets), 3))
        for i, asset in enumerate(cat_assets):
            status = asset["status"]
            css    = {"IN": "ac-in", "WATCH": "ac-watch", "OUT": "ac-out"}.get(status, "ac-out")
            sc_col = {"IN": COLORS["green"], "WATCH": COLORS["gold"], "OUT": "#4a2a2a"}.get(status, COLORS["gray"])
            bg_col = {"IN": "#0a2a1a", "WATCH": "#1a1a0a", "OUT": "#0d0d0d"}.get(status, "#0d1220")
            with cols[i % 3]:
                st.markdown(
                    f'<div style="background:{bg_col}; border:1px solid {sc_col}33; '
                    f'border-radius:8px; padding:10px 14px; margin-bottom:10px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="color:#00d4ff; font-family:\'JetBrains Mono\',monospace; font-weight:700; font-size:0.9rem;">{asset["etf"]}</span>'
                    f'<span class="{css}">{status}</span>'
                    f'</div>'
                    f'<div style="color:#a0b4cc; font-size:0.78rem; margin-top:4px;">{asset["label"]}</div>'
                    f'<div style="color:#6b8099; font-size:0.72rem; margin-top:2px;">{asset["description"]}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
        st.markdown("")


# ═══════════════════════════════════════════════════════
#  TAB 3 — AI ANALYST
# ═══════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🤖 QuantEdge Analyst — Cougar Global Methodology")
    st.markdown(
        '<div style="color:#6b8099; font-size:0.88rem; margin-bottom:20px;">'
        'PMPT-based institutional analysis. Macro regime → tactical allocation → downside risk framing. '
        'Powered by Claude, persona modeled on Cougar Global\'s investment process.'
        '</div>', unsafe_allow_html=True,
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_analysis = st.button("🔍 Run PMPT Analysis", use_container_width=True, type="primary")

    if run_analysis:
        if not analyst.available:
            st.warning("⚠️ AI analyst offline. Showing static Cougar Global rules-based analysis.")
        with st.spinner("Running Cougar Global 3-step analysis (regime → allocation → risk)..."):
            port_dict = portfolio_df.to_dict(orient="records")
            result = analyst.analyze_portfolio(
                portfolio_data={"holdings": port_dict, "summary": summary},
                risk_metrics=risk_metrics if st.session_state.risk_metrics else {},
                macro_data=macro_data,
                investable_cash=investable_cash,
                allocation_plan=allocation_plan,
                regime_assessment=algo_regime,
            )
            st.session_state.analysis_result = result

    if st.session_state.analysis_result:
        st.markdown(
            f'<div class="analyst-response">{st.session_state.analysis_result}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d1a2e,#0a1525); border:1px solid #1e4a6b;
             border-radius:12px; padding:32px; text-align:center; margin-top:20px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">🧠</div>
            <div style="color:#00d4ff; font-size:1.15rem; font-weight:600; margin-bottom:8px;">
                QuantEdge Analyst — Cougar Global Mode
            </div>
            <div style="color:#6b8099; font-size:0.9rem; max-width:500px; margin:0 auto;">
                Click <strong style="color:#e0e6f0;">Run PMPT Analysis</strong> for a full Cougar Global-style report:
                macro regime assessment → PMPT risk profile (Probability of Loss) →
                tactical deployment plan → bear market contingency → 10-year compounding roadmap.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Watchlist Quick Takes ──
    st.markdown("---")
    st.markdown("#### 🎯 Watchlist — Regime-Conditional Quick Takes")

    watch_tickers = tuple(w["ticker"] for w in WATCHLIST_RECOMMENDATIONS)
    watch_prices  = load_market_data(watch_tickers)

    for item in WATCHLIST_RECOMMENDATIONS:
        ticker = item["ticker"]
        pd_    = watch_prices.get(ticker, {})
        price  = pd_.get("price")
        chg    = pd_.get("change_pct")

        label = f"{'📦' if item['type']=='ETF' else '📈'} {ticker}"
        if price and chg is not None:
            label += f"  ${price:.2f}  ({chg:+.2f}%)"

        with st.expander(label, expanded=False):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                if price:
                    st.metric("Price", f"${price:.2f}", delta=f"{chg:+.2f}%" if chg is not None else None)
                st.markdown(f"**Type:** {item['type']}")
            with col_b:
                st.markdown(f"**Rationale:** {item['rationale']}")
                if st.button(f"Regime Take on {ticker}", key=f"quick_{ticker}"):
                    with st.spinner(f"Analyzing {ticker} for {active_regime} regime..."):
                        take = analyst.quick_ticker_take(ticker, price or 0, chg or 0)
                        st.markdown(
                            f'<div style="background:#0d1a2e; border:1px solid #1e2d4a; '
                            f'border-left:3px solid #00d4ff; border-radius:6px; padding:14px; '
                            f'font-size:0.88rem; line-height:1.6;">{take}</div>',
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════
#  TAB 4 — ROADMAP (Monte Carlo + POL over time)
# ═══════════════════════════════════════════════════════

with tab4:
    st.markdown("### 📈 Investment Roadmap — Monte Carlo Simulation")
    st.markdown(
        '<div style="color:#6b8099; font-size:0.88rem; margin-bottom:16px;">'
        '500-path Monte Carlo simulation with Probability of Loss bands. '
        'Bear floor and bull ceiling shown alongside compounding roadmap.'
        '</div>', unsafe_allow_html=True,
    )

    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    with mc_col1: mc_years   = st.slider("Time Horizon (Years)", 5, 30, 10)
    with mc_col2: mc_return  = st.slider("Expected Ann. Return (%)", 8, 25, 14) / 100
    with mc_col3: mc_vol     = st.slider("Expected Volatility (%)", 10, 35, 20) / 100
    with mc_col4: mc_contrib = st.number_input("Annual Contribution ($)", 0, 7000, int(annual_contribution), step=500)

    initial_value = (summary["total_value"] or roth_base) + investable_cash

    with st.spinner("Running 500-path Monte Carlo with PMPT bootstrapping..."):
        mc = monte_carlo_simulation(
            initial_value=initial_value, annual_return_mean=mc_return,
            annual_vol=mc_vol, years=mc_years, simulations=500,
            annual_contribution=mc_contrib,
        )

    years_arr = mc["years"]

    # ── Growth Chart ──
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=years_arr + years_arr[::-1], y=mc["p95"] + mc["p5"][::-1],
        fill="toself", fillcolor="rgba(0,212,255,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI",
    ))
    fig_mc.add_trace(go.Scatter(
        x=years_arr + years_arr[::-1], y=mc["p75"] + mc["p25"][::-1],
        fill="toself", fillcolor="rgba(0,212,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="50% CI",
    ))
    for label, key, color, width, dash in [
        ("Bear Floor (5th)",   "p5",  COLORS["red"],   1.5, "dot"),
        ("Base (25th)",        "p25", COLORS["gold"],  1.5, "dash"),
        ("Median",             "p50", COLORS["blue"],  3.0, "solid"),
        ("Bull (75th)",        "p75", COLORS["green"], 1.5, "dash"),
        ("Bull Ceiling (95th)","p95", COLORS["green"], 1.5, "dot"),
    ]:
        fig_mc.add_trace(go.Scatter(x=years_arr, y=mc[key], name=label,
                                     line=dict(color=color, width=width, dash=dash)))

    fig_mc.add_annotation(
        x=mc_years, y=mc["p50"][-1],
        text=f"Median: {fmt_currency(mc['p50'][-1])}",
        showarrow=True, arrowhead=2,
        font=dict(color=COLORS["blue"], size=12),
        bgcolor="#0d1a2e", bordercolor=COLORS["blue"], arrowcolor=COLORS["blue"],
    )
    fig_mc.update_layout(**PLOT_LAYOUT, height=420,
                          title=f"Portfolio Growth — {mc_years}-Year Monte Carlo",
                          xaxis_title="Years", yaxis_title="Value ($)", yaxis_tickformat="$,.0f",
                          hovermode="x unified")
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Probability of Loss Over Time ──
    st.markdown("#### 🛡️ Probability of Loss by Year (PMPT Core Metric)")
    pol_vals = mc["prob_of_loss_by_year"]
    pol_colors_list = [pol_color(p) for p in pol_vals]

    fig_pol = go.Figure()
    fig_pol.add_trace(go.Bar(
        x=years_arr, y=pol_vals,
        marker_color=pol_colors_list,
        text=[f"{p:.1f}%" for p in pol_vals],
        textposition="outside",
        textfont=dict(color="#a0b4cc", size=11),
        name="Probability of Loss",
    ))
    fig_pol.add_hline(y=20, line_dash="dash", line_color=COLORS["gold"],
                       annotation_text="Target POL threshold: 20%",
                       annotation_font_color=COLORS["gold"])
    fig_pol.update_layout(**PLOT_LAYOUT, height=320,
                           title="Probability of Portfolio Loss — by Year Horizon",
                           xaxis_title="Year", yaxis_title="Probability of Loss (%)",
                           yaxis_range=[0, max(pol_vals) * 1.3 if pol_vals else 50])
    st.plotly_chart(fig_pol, use_container_width=True)

    # ── Summary Stats ──
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Starting Value", fmt_currency(initial_value))
    with s2:
        st.metric(f"Median ({mc_years}yr)", fmt_currency(mc["final_p50"]),
                  delta=fmt_pct((mc["final_p50"]/initial_value-1)*100) if initial_value else None)
    with s3: st.metric(f"🐂 Bull Ceiling", fmt_currency(mc["bull_ceiling"]))
    with s4: st.metric(f"🐻 Bear Floor", fmt_currency(mc["bear_floor"]))

    st.markdown("---")

    # ── Milestones ──
    st.markdown("#### 📅 Year-by-Year Milestones")
    milestones = []
    for i, yr in enumerate(years_arr):
        if yr in [1,3,5,7,10,15,20,25,30] and yr <= mc_years:
            milestones.append({
                "Year": yr,
                "Prob Loss (%)": f"{mc['prob_of_loss_by_year'][i]:.1f}%",
                "Bear (5th)":    fmt_currency(mc["p5"][i]),
                "Base (25th)":   fmt_currency(mc["p25"][i]),
                "Median":        fmt_currency(mc["p50"][i]),
                "Bull (75th)":   fmt_currency(mc["p75"][i]),
                "Ceiling (95th)":fmt_currency(mc["p95"][i]),
            })
    if milestones:
        st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Roth Tax Advantage ──
    st.markdown("#### 💎 Roth IRA Tax-Free Compounding")
    roth_terminal    = mc["final_p50"]
    taxable_terminal = initial_value + (roth_terminal - initial_value) * 0.80
    tc1, tc2, tc3 = st.columns(3)
    with tc1: st.metric("Roth IRA (Tax-Free)", fmt_currency(roth_terminal))
    with tc2: st.metric("Taxable Equiv. (20% LTCG)", fmt_currency(taxable_terminal))
    with tc3: st.metric("Roth Tax Advantage", fmt_currency(roth_terminal - taxable_terminal))


# ═══════════════════════════════════════════════════════
#  TAB 5 — MARKET INTEL
# ═══════════════════════════════════════════════════════

with tab5:
    st.markdown("### 🌍 Market Intelligence — Macro Regime Monitor")

    if st.button("🧠 AI Macro Regime Commentary", type="primary"):
        with st.spinner("Analyst reading macro for regime signals..."):
            commentary = analyst.market_commentary(macro_data)
            st.markdown(f'<div class="analyst-response">{commentary}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📡 Live Macro Indicators — Regime Signals")

    if macro_data:
        macro_cols = st.columns(4)
        for i, item in enumerate(macro_data):
            with macro_cols[i % 4]:
                chg_color = COLORS["green"] if item["change"] >= 0 else COLORS["red"]
                chg_arrow = "▲" if item["change"] >= 0 else "▼"
                # Regime implication label
                regime_impl = ""
                if item["label"] == "VIX":
                    regime_impl = "🐂 Bullish" if item["price"] < 16 else ("🐻 Bearish" if item["price"] > 24 else "⚖️ Neutral")
                elif item["label"] == "10-Yr Yield":
                    regime_impl = "⚖️ Neutral" if item["price"] < 5 else "🐻 Restrictive"
                elif item["label"] == "Gold":
                    regime_impl = "🐻 Flight-safety" if item["change"] > 0.5 else "🐂 Risk-on"

                st.markdown(
                    f'<div style="background:#0d1220; border:1px solid #1e2d4a; border-radius:8px; '
                    f'padding:14px; text-align:center; margin-bottom:12px;">'
                    f'<div style="color:#6b8099; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">{item["label"]}</div>'
                    f'<div style="color:#e0e6f0; font-size:1.25rem; font-weight:700; margin:4px 0;">{item["price"]:.2f}</div>'
                    f'<div style="color:{chg_color}; font-size:0.85rem; font-weight:600;">{chg_arrow} {item["change"]:+.2f}%</div>'
                    + (f'<div style="color:#6b8099; font-size:0.72rem; margin-top:4px;">{regime_impl}</div>' if regime_impl else "")
                    + f'</div>', unsafe_allow_html=True,
                )
    else:
        st.info("Live macro data unavailable — check internet connection.")

    st.markdown("---")
    st.markdown("#### 🔥 Portfolio Universe Movers")

    all_tickers_list = [h["ticker"] for h in DEFAULT_ROTH_HOLDINGS] + [w["ticker"] for w in WATCHLIST_RECOMMENDATIONS]
    movers = fetch_portfolio_movers(list(set(all_tickers_list)))

    if movers:
        movers_df = pd.DataFrame(movers).dropna(subset=["change"]).sort_values("change", ascending=True)
        bar_colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in movers_df["change"]]
        fig_movers = go.Figure(go.Bar(
            x=movers_df["change"], y=movers_df["ticker"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{c:+.2f}%" for c in movers_df["change"]],
            textposition="outside", textfont=dict(size=11, color="#a0b4cc"),
        ))
        fig_movers.update_layout(**PLOT_LAYOUT, height=350,
                                  title="Today's Performance — Portfolio Universe",
                                  xaxis_title="Day Change (%)", xaxis_tickformat="+.1f%",
                                  yaxis_tickfont=dict(family="JetBrains Mono", size=12))
        st.plotly_chart(fig_movers, use_container_width=True)

    st.markdown("---")

    # ── Cougar Global Factor Dashboard ──
    st.markdown("#### 📐 Factor ETF Monitor — Regime Context")
    factor_tickers = ("MTUM", "QUAL", "USMV", "VTV", "IWF", "XMMO")
    factor_prices  = load_market_data(factor_tickers)
    factor_data = [
        {"Factor": "Momentum",       "ETF": "MTUM", "Regime": "Bull"},
        {"Factor": "Quality",        "ETF": "QUAL", "Regime": "Base"},
        {"Factor": "Min Volatility", "ETF": "USMV", "Regime": "Bear"},
        {"Factor": "Value",          "ETF": "VTV",  "Regime": "Base"},
        {"Factor": "Growth",         "ETF": "IWF",  "Regime": "Bull"},
        {"Factor": "Mid Momentum",   "ETF": "XMMO", "Regime": "Bull"},
    ]
    f_cols = st.columns(3)
    for i, f in enumerate(factor_data):
        with f_cols[i % 3]:
            pd_  = factor_prices.get(f["ETF"], {})
            price = pd_.get("price")
            chg   = pd_.get("change_pct")
            chg_color = COLORS["green"] if (chg or 0) >= 0 else COLORS["red"]
            regime_match = f["Regime"] == active_regime.title()
            badge_color  = COLORS["green"] if regime_match else COLORS["gray"]
            st.markdown(
                f'<div style="background:#0d1220; border:1px solid {"#00ff8844" if regime_match else "#1e2d4a"}; '
                f'border-radius:8px; padding:12px 16px; margin-bottom:10px;">'
                f'<div style="display:flex; justify-content:space-between;">'
                f'<span style="color:#00d4ff; font-family:\'JetBrains Mono\',monospace; font-weight:700;">{f["ETF"]}</span>'
                f'<span style="color:{badge_color}; font-size:0.72rem;">✓ {f["Regime"]}</span>'
                f'</div>'
                f'<div style="color:#a0b4cc; font-size:0.78rem; margin-top:2px;">{f["Factor"]}</div>'
                f'<div style="color:#e0e6f0; font-size:1rem; font-weight:600; margin-top:4px;">'
                f'{"$"+f"{price:.2f}" if price else "N/A"}</div>'
                f'<div style="color:{chg_color}; font-size:0.82rem;">{chg:+.2f}% today</div>'
                f'</div>', unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════
#  TAB 6 — CHAT
# ═══════════════════════════════════════════════════════

with tab6:
    st.markdown("### 💬 Chat with QuantEdge Analyst")
    st.markdown(
        '<div style="color:#6b8099; font-size:0.88rem; margin-bottom:16px;">'
        'Analyst stays in Cougar Global PMPT persona: probability of loss, macro regime, '
        'tactical rotation. Ask anything about your portfolio or methodology.'
        '</div>', unsafe_allow_html=True,
    )

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="background:#0d1a2e; border:1px solid #1e2d4a; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px; border-left:3px solid #ffd700;">'
                    f'<span style="color:#ffd700; font-size:0.75rem; font-weight:600;">YOU</span><br>'
                    f'<span style="color:#e0e6f0;">{msg["content"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#0a1525; border:1px solid #1e4a6b; border-radius:8px; '
                    f'padding:12px 16px; margin-bottom:8px; border-left:3px solid #00d4ff;">'
                    f'<span style="color:#00d4ff; font-size:0.75rem; font-weight:600;">⚡ QUANTEDGE ANALYST</span><br>'
                    f'<span style="color:#e0e6f0; font-size:0.9rem; line-height:1.6;">{msg["content"]}</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.markdown("**Quick Prompts — Cougar Global Topics:**")
    qp_cols = st.columns(3)
    quick_prompts = [
        "What's the current macro regime and what does it mean for my allocation?",
        "Explain Probability of Loss vs Sharpe ratio — why does Cougar Global prefer POL?",
        "If VIX spikes to 30 tomorrow, what should I rotate into?",
        "Walk me through the bear regime portfolio — why TLT and gold over equities?",
        "What's my portfolio's biggest downside risk right now?",
        "How does tactical allocation help survive bear markets vs buy-and-hold?",
    ]
    for i, qp in enumerate(quick_prompts):
        with qp_cols[i % 3]:
            if st.button(qp[:45] + "...", key=f"qp_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": qp})
                with st.spinner("Analyst composing regime-aware response..."):
                    port_context = (
                        f"Portfolio: {portfolio_df[['Ticker','Value','Weight%']].to_string()}\n"
                        f"Risk metrics: {risk_metrics}\n"
                        f"Current macro regime: {active_regime} ({algo_regime['confidence']}% confidence)\n"
                        f"Regime signals: {', '.join(algo_regime.get('signal_log', []))}"
                    )
                    response = analyst.chat(qp, portfolio_context=port_context)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    st.markdown("")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask the analyst...",
            placeholder="e.g., 'We're in a base regime — how do I reduce my Probability of Loss below 15%?'",
            height=80, label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([1, 5])
        with col_send:
            send = st.form_submit_button("Send →", use_container_width=True, type="primary")
        with col_clear:
            if st.form_submit_button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("QuantEdge Analyst thinking..."):
            port_context = (
                f"Holdings: {portfolio_df[['Ticker','Value','Weight%']].to_string()}\n"
                f"Risk metrics: {risk_metrics}\n"
                f"Investable cash: ${investable_cash:,.0f}\n"
                f"Current macro regime: {active_regime} ({algo_regime['confidence']}% confidence)\n"
                f"Rebalancing signals: {[s['action']+' '+s['ticker'] for s in rebalancing_signals[:5]]}"
            )
            response = analyst.chat(user_input, portfolio_context=port_context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if not analyst.available:
        st.info("💡 Add your ANTHROPIC_API_KEY in the sidebar to enable live analyst chat.")
