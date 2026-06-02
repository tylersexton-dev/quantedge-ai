"""
portfolio_engine.py
QuantEdge AI — Portfolio calculation engine
Handles live data fetching, risk metrics, Monte Carlo simulation, and sector analysis.

Incorporates Cougar Global Investments methodology:
- Tactical asset allocation across 20 global asset classes
- Probability of Loss as primary risk metric (Post Modern Portfolio Theory)
- Three-scenario macro framework: Bull / Base / Bear
- Downside-first risk framing; bootstrapping-style simulation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Tyler's pre-populated portfolio
# ─────────────────────────────────────────────

DEFAULT_ROTH_HOLDINGS = [
    {"ticker": "VTI",   "shares": 0.35, "account": "Roth IRA", "type": "ETF",   "name": "Vanguard Total Market ETF"},
    {"ticker": "QQQ",   "shares": 0.15, "account": "Roth IRA", "type": "ETF",   "name": "Invesco QQQ Trust"},
    {"ticker": "SCHG",  "shares": 0.20, "account": "Roth IRA", "type": "ETF",   "name": "Schwab US Large-Cap Growth ETF"},
    {"ticker": "NVDA",  "shares": 0.08, "account": "Roth IRA", "type": "Stock", "name": "NVIDIA Corporation"},
    {"ticker": "MSFT",  "shares": 0.05, "account": "Roth IRA", "type": "Stock", "name": "Microsoft Corporation"},
]

DEFAULT_INVESTABLE_CASH = 2500.0  # midpoint of $2k–$3k

BENCHMARK_TICKER = "SPY"
RISK_FREE_RATE = 0.053  # ~5.3% 10-yr Treasury yield (2024)

SECTOR_MAP = {
    "VTI":  "Broad Market",
    "QQQ":  "Technology",
    "SCHG": "Growth",
    "VUG":  "Growth",
    "ARKK": "Innovation",
    "NVDA": "Technology",
    "MSFT": "Technology",
    "AAPL": "Technology",
    "AMZN": "Consumer Discretionary",
    "GOOGL":"Technology",
    "META": "Technology",
    "TSLA": "Consumer Discretionary",
    "AMD":  "Technology",
    "PLTR": "Technology",
    "SMCI": "Technology",
    "SPY":  "Broad Market",
    "IWF":  "Growth",
    "VGT":  "Technology",
    "SOXX": "Semiconductors",
    "XMMO": "Momentum",
    "MTUM": "Momentum",
}

WATCHLIST_RECOMMENDATIONS = [
    {"ticker": "VUG",  "type": "ETF",   "rationale": "Vanguard Growth ETF — low-cost pure large-cap growth exposure, ideal Roth core sleeve"},
    {"ticker": "SCHG", "type": "ETF",   "rationale": "Schwab Large-Cap Growth — lowest ER growth ETF, mirrors QQQ with broader diversification"},
    {"ticker": "XMMO", "type": "ETF",   "rationale": "Invesco S&P MidCap Momentum — quant momentum factor, historically outperforms in bull cycles"},
    {"ticker": "MTUM", "type": "ETF",   "rationale": "iShares MSCI USA Momentum Factor — systematic momentum, top Sharpe ratio in factor space"},
    {"ticker": "SOXX", "type": "ETF",   "rationale": "iShares Semiconductor ETF — AI infrastructure cycle play, secular tailwind"},
    {"ticker": "NVDA", "type": "Stock", "rationale": "NVIDIA — dominant AI compute platform, pricing power moat, best risk-adjusted growth"},
    {"ticker": "PLTR", "type": "Stock", "rationale": "Palantir — AI/ML government & enterprise SaaS, accelerating revenue, Rule of 40+ metrics"},
    {"ticker": "AMZN", "type": "Stock", "rationale": "Amazon — AWS margin expansion + ad business undervalued, quality compounder"},
]


# ─────────────────────────────────────────────
#  Data Fetching
# ─────────────────────────────────────────────

def fetch_price(ticker: str) -> dict:
    """Fetch current price + basic info for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="2d")
        if hist.empty:
            return {"price": None, "prev_close": None, "change_pct": None}
        price = float(hist["Close"].iloc[-1])
        prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        return {
            "price": price,
            "prev_close": prev,
            "change_pct": (price - prev) / prev * 100 if prev else 0,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "beta": info.get("beta"),
            "name": info.get("longName", ticker),
        }
    except Exception:
        return {"price": None, "prev_close": None, "change_pct": None}


def fetch_prices_batch(tickers: list) -> dict:
    """Batch-fetch prices for a list of tickers."""
    results = {}
    try:
        data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        for ticker in tickers:
            try:
                col = closes[ticker] if ticker in closes.columns else closes
                col = col.dropna()
                if len(col) >= 2:
                    price = float(col.iloc[-1])
                    prev  = float(col.iloc[-2])
                    results[ticker] = {
                        "price": price,
                        "change_pct": (price - prev) / prev * 100,
                    }
                elif len(col) == 1:
                    results[ticker] = {"price": float(col.iloc[-1]), "change_pct": 0.0}
                else:
                    results[ticker] = {"price": None, "change_pct": None}
            except Exception:
                results[ticker] = {"price": None, "change_pct": None}
    except Exception:
        for t in tickers:
            results[t] = {"price": None, "change_pct": None}
    return results


def fetch_historical_returns(tickers: list, period_days: int = 365) -> pd.DataFrame:
    """Return daily percentage returns for a list of tickers."""
    end   = datetime.today()
    start = end - timedelta(days=period_days)
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data
        return closes.pct_change().dropna()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  Portfolio Valuation
# ─────────────────────────────────────────────

def build_portfolio(holdings: list, prices: dict) -> pd.DataFrame:
    """Build a portfolio DataFrame with current values."""
    rows = []
    for h in holdings:
        ticker = h["ticker"]
        price_info = prices.get(ticker, {})
        price = price_info.get("price")
        value = (h["shares"] * price) if price else None
        rows.append({
            "Ticker":    ticker,
            "Name":      h.get("name", ticker),
            "Shares":    h["shares"],
            "Price":     price,
            "Value":     value,
            "Change%":   price_info.get("change_pct"),
            "Account":   h.get("account", "Taxable"),
            "Type":      h.get("type", "ETF"),
            "Sector":    SECTOR_MAP.get(ticker, "Other"),
        })
    df = pd.DataFrame(rows)
    total = df["Value"].sum()
    df["Weight%"] = (df["Value"] / total * 100).round(2) if total else 0
    return df


def portfolio_summary(portfolio_df: pd.DataFrame) -> dict:
    """Return high-level portfolio statistics."""
    df = portfolio_df.dropna(subset=["Value"])
    total_value  = df["Value"].sum()
    total_gain_  = 0  # would need cost basis; skip for now
    return {
        "total_value": total_value,
        "num_positions": len(df),
        "etf_weight": df[df["Type"] == "ETF"]["Value"].sum() / total_value * 100 if total_value else 0,
        "stock_weight": df[df["Type"] == "Stock"]["Value"].sum() / total_value * 100 if total_value else 0,
    }


# ─────────────────────────────────────────────
#  Risk Metrics
# ─────────────────────────────────────────────

def compute_portfolio_returns(portfolio_df: pd.DataFrame, period_days: int = 252) -> pd.Series:
    """Compute weighted daily returns for the portfolio."""
    tickers = portfolio_df["Ticker"].tolist()
    weights = portfolio_df.set_index("Ticker")["Weight%"] / 100
    returns = fetch_historical_returns(tickers, period_days=period_days + 30)
    if returns.empty:
        return pd.Series(dtype=float)
    available = [t for t in tickers if t in returns.columns]
    w = weights[available]
    w = w / w.sum()
    port_ret = (returns[available] * w).sum(axis=1)
    return port_ret


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns.mean() * 252 - rf
    vol    = returns.std() * np.sqrt(252)
    return round(excess / vol, 3)


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if returns.empty:
        return 0.0
    downside = returns[returns < 0].std() * np.sqrt(252)
    if downside == 0:
        return 0.0
    excess = returns.mean() * 252 - rf
    return round(excess / downside, 3)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak-to-trough."""
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return round(drawdown.min() * 100, 2)


def portfolio_beta(portfolio_returns: pd.Series, period_days: int = 252) -> float:
    """Beta vs. S&P 500."""
    try:
        bench = fetch_historical_returns([BENCHMARK_TICKER], period_days=period_days + 30)
        if bench.empty or BENCHMARK_TICKER not in bench.columns:
            return 1.0
        bench_ret = bench[BENCHMARK_TICKER]
        aligned = pd.concat([portfolio_returns, bench_ret], axis=1).dropna()
        aligned.columns = ["port", "bench"]
        cov  = aligned.cov().iloc[0, 1]
        var  = aligned["bench"].var()
        return round(cov / var, 3) if var != 0 else 1.0
    except Exception:
        return 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility (std of returns)."""
    if returns.empty:
        return 0.0
    return round(returns.std() * np.sqrt(252) * 100, 2)


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """
    Annualized downside deviation relative to a Minimum Acceptable Return (MAR).
    Core metric in Post Modern Portfolio Theory (PMPT) — Cougar Global's framework.
    Default MAR = 0 (i.e., probability of any loss).
    """
    if returns.empty:
        return 0.0
    daily_mar = mar / 252
    downside = returns[returns < daily_mar] - daily_mar
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(252) * 100
    return round(dd, 2)


def probability_of_loss_historical(returns: pd.Series, horizon_days: int = 252) -> float:
    """
    Bootstrapped Probability of Loss over a given horizon.
    Cougar Global's primary risk metric (PMPT).
    Resamples historical daily returns to simulate horizon-length paths,
    returns the fraction of paths that end below starting value.
    """
    if returns.empty or len(returns) < 20:
        return None
    rng = np.random.default_rng(42)
    n_sims = 2000
    # Bootstrap: draw horizon_days returns with replacement, compound
    idx = rng.integers(0, len(returns), size=(n_sims, horizon_days))
    sampled = returns.values[idx]
    path_returns = (1 + sampled).prod(axis=1) - 1
    pol = float(np.mean(path_returns < 0) * 100)
    return round(pol, 1)


def compute_risk_metrics(portfolio_df: pd.DataFrame) -> dict:
    """Compute all risk metrics for the portfolio, including PMPT metrics."""
    port_ret = compute_portfolio_returns(portfolio_df)
    if port_ret.empty:
        return {
            "sharpe": "N/A", "sortino": "N/A", "beta": "N/A",
            "max_drawdown": "N/A", "volatility": "N/A",
            "annualized_return": "N/A",
            "downside_deviation": "N/A",
            "prob_of_loss_1yr": "N/A",
            "prob_of_loss_3yr": "N/A",
        }
    ann_return = round((1 + port_ret.mean()) ** 252 - 1, 4) * 100
    pol_1yr = probability_of_loss_historical(port_ret, horizon_days=252)
    pol_3yr = probability_of_loss_historical(port_ret, horizon_days=756)
    return {
        "sharpe":             sharpe_ratio(port_ret),
        "sortino":            sortino_ratio(port_ret),
        "beta":               portfolio_beta(port_ret),
        "max_drawdown":       max_drawdown(port_ret),
        "volatility":         annualized_volatility(port_ret),
        "annualized_return":  round(ann_return, 2),
        "downside_deviation": downside_deviation(port_ret),
        "prob_of_loss_1yr":   pol_1yr,
        "prob_of_loss_3yr":   pol_3yr,
    }


# ─────────────────────────────────────────────
#  Monte Carlo Simulation
# ─────────────────────────────────────────────

def monte_carlo_simulation(
    initial_value: float,
    annual_return_mean: float = 0.12,
    annual_vol: float = 0.18,
    years: int = 10,
    simulations: int = 500,
    annual_contribution: float = 0.0,
) -> dict:
    """
    Monte Carlo simulation for portfolio growth.
    Returns percentile bands (5th, 25th, 50th, 75th, 95th).
    """
    dt = 1 / 252  # daily steps
    steps = int(years * 252)
    daily_mean = annual_return_mean / 252
    daily_vol  = annual_vol / np.sqrt(252)
    daily_contrib = annual_contribution / 252

    paths = np.zeros((simulations, steps + 1))
    paths[:, 0] = initial_value

    rng = np.random.default_rng(42)
    for t in range(1, steps + 1):
        shocks = rng.normal(daily_mean, daily_vol, simulations)
        paths[:, t] = paths[:, t - 1] * (1 + shocks) + daily_contrib

    # Sample at yearly intervals
    yearly_idx = [int(i * 252) for i in range(years + 1)]
    yearly_paths = paths[:, yearly_idx]

    # Probability of Loss at each year (PMPT — fraction of paths below initial)
    pol_by_year = [
        float(np.mean(yearly_paths[:, i] < initial_value) * 100)
        for i in range(years + 1)
    ]

    return {
        "years":    list(range(years + 1)),
        "p5":       np.percentile(yearly_paths, 5,  axis=0).tolist(),
        "p25":      np.percentile(yearly_paths, 25, axis=0).tolist(),
        "p50":      np.percentile(yearly_paths, 50, axis=0).tolist(),
        "p75":      np.percentile(yearly_paths, 75, axis=0).tolist(),
        "p95":      np.percentile(yearly_paths, 95, axis=0).tolist(),
        "final_p50": float(np.percentile(yearly_paths[:, -1], 50)),
        "final_p95": float(np.percentile(yearly_paths[:, -1], 95)),
        "final_p5":  float(np.percentile(yearly_paths[:, -1],  5)),
        "prob_of_loss_by_year": pol_by_year,
        "prob_of_loss_final":   pol_by_year[-1],
        # Bear-case floor: 5th percentile final value
        "bear_floor":  float(np.percentile(yearly_paths[:, -1],  5)),
        "bull_ceiling": float(np.percentile(yearly_paths[:, -1], 95)),
        "all_paths_final": yearly_paths[:, -1].tolist(),
    }


# ─────────────────────────────────────────────
#  Macro / Market Data
# ─────────────────────────────────────────────

MACRO_TICKERS = {
    "S&P 500":    "^GSPC",
    "NASDAQ 100": "^NDX",
    "Russell 2000": "^RUT",
    "VIX":        "^VIX",
    "10-Yr Yield": "^TNX",
    "Gold":       "GLD",
    "Oil (WTI)":  "USO",
    "Bitcoin":    "BTC-USD",
}


def fetch_macro_data() -> list:
    """Fetch macro indicators."""
    results = []
    for label, ticker in MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty or len(hist) < 2:
                continue
            price   = float(hist["Close"].iloc[-1])
            prev    = float(hist["Close"].iloc[-2])
            chg_pct = (price - prev) / prev * 100
            results.append({
                "label":    label,
                "ticker":   ticker,
                "price":    price,
                "change":   chg_pct,
            })
        except Exception:
            continue
    return results


def fetch_portfolio_movers(tickers: list) -> list:
    """Get top movers for portfolio tickers."""
    prices = fetch_prices_batch(tickers)
    movers = []
    for ticker, data in prices.items():
        if data.get("change_pct") is not None:
            movers.append({
                "ticker": ticker,
                "price":  data["price"],
                "change": data["change_pct"],
            })
    return sorted(movers, key=lambda x: abs(x["change"] or 0), reverse=True)


# ─────────────────────────────────────────────
#  Allocation Recommendations
# ─────────────────────────────────────────────

def generate_allocation_plan(cash: float) -> list:
    """
    Generate a ranked allocation plan for investable cash.
    Follows quant/momentum factor principles.
    """
    if cash <= 0:
        return []

    plans = [
        {
            "rank": 1,
            "ticker": "SCHG",
            "name": "Schwab US Large-Cap Growth ETF",
            "alloc_pct": 35,
            "amount": round(cash * 0.35, 2),
            "rationale": "Core growth sleeve. Lowest ER (~0.04%) in large-cap growth. Captures mega-cap tech momentum with diversification.",
            "priority": "Buy First",
        },
        {
            "rank": 2,
            "ticker": "XMMO",
            "name": "Invesco S&P MidCap Momentum ETF",
            "alloc_pct": 25,
            "amount": round(cash * 0.25, 2),
            "rationale": "Systematic momentum factor exposure. Mid-cap momentum has highest risk-adjusted alpha historically. Quant-driven rebalancing.",
            "priority": "Buy First",
        },
        {
            "rank": 3,
            "ticker": "SOXX",
            "name": "iShares Semiconductor ETF",
            "alloc_pct": 20,
            "amount": round(cash * 0.20, 2),
            "rationale": "Sector concentration bet on AI infrastructure cycle. Semiconductors are the 'picks and shovels' of the AI build-out.",
            "priority": "Buy Next",
        },
        {
            "rank": 4,
            "ticker": "NVDA",
            "name": "NVIDIA Corporation",
            "alloc_pct": 12,
            "amount": round(cash * 0.12, 2),
            "rationale": "Highest-conviction individual growth name. Data center revenue compounding at 100%+ YoY. AI infrastructure monopoly.",
            "priority": "Buy Next",
        },
        {
            "rank": 5,
            "ticker": "PLTR",
            "name": "Palantir Technologies",
            "alloc_pct": 8,
            "amount": round(cash * 0.08, 2),
            "rationale": "AI/ML government & commercial SaaS. Rule of 40 metrics accelerating. High-alpha satellite position.",
            "priority": "Optional",
        },
    ]
    return plans


def rebalancing_alerts(portfolio_df: pd.DataFrame) -> list:
    """Check if portfolio is outside target allocation bands."""
    alerts = []
    etf_weight   = portfolio_df[portfolio_df["Type"] == "ETF"]["Weight%"].sum()
    stock_weight = portfolio_df[portfolio_df["Type"] == "Stock"]["Weight%"].sum()

    if etf_weight < 60:
        alerts.append({"type": "warning", "msg": f"ETF sleeve at {etf_weight:.1f}% — below 70% target. Consider trimming individual stocks."})
    if stock_weight > 35:
        alerts.append({"type": "warning", "msg": f"Individual stock exposure at {stock_weight:.1f}% — above 30% target. Concentration risk elevated."})
    if etf_weight > 85:
        alerts.append({"type": "info", "msg": f"ETF sleeve at {etf_weight:.1f}% — fully diversified. Add growth ETF or individual name to optimize alpha."})

    # Check for over-concentration in single position
    for _, row in portfolio_df.iterrows():
        if row["Weight%"] > 30:
            alerts.append({"type": "danger", "msg": f"{row['Ticker']} at {row['Weight%']:.1f}% — single-position concentration risk. Consider trimming."})
    return alerts


# ═════════════════════════════════════════════════════════════
#  COUGAR GLOBAL INVESTMENTS METHODOLOGY
#  Tactical Asset Allocation · 20 Asset Class Framework · PMPT
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
#  20 Global Asset Class Universe
# ─────────────────────────────────────────────

ASSET_CLASS_UNIVERSE: list[dict] = [
    # ── US Equity ──
    {"id": "us_lc_growth",   "label": "US Large Cap Growth",    "etf": "IWF",  "category": "US Equity",       "description": "Russell 1000 Growth — mega-cap tech & growth"},
    {"id": "us_lc_value",    "label": "US Large Cap Value",     "etf": "IWD",  "category": "US Equity",       "description": "Russell 1000 Value — financials, energy, healthcare"},
    {"id": "us_sc_growth",   "label": "US Small Cap Growth",    "etf": "IJT",  "category": "US Equity",       "description": "S&P 600 Small Cap Growth — high beta domestic"},
    {"id": "us_sc_value",    "label": "US Small Cap Value",     "etf": "IJS",  "category": "US Equity",       "description": "S&P 600 Small Cap Value — cyclical value exposure"},
    {"id": "us_midcap",      "label": "US Mid Cap",             "etf": "IJH",  "category": "US Equity",       "description": "S&P 400 Mid Cap — balanced growth/value blend"},
    {"id": "us_momentum",    "label": "US Momentum",            "etf": "MTUM", "category": "US Equity",       "description": "MSCI USA Momentum Factor — systematic factor tilt"},
    # ── International Equity ──
    {"id": "intl_dev",       "label": "International Developed","etf": "EFA",  "category": "Intl Equity",     "description": "MSCI EAFE — Europe, Australasia, Far East"},
    {"id": "intl_sc",        "label": "International Small Cap","etf": "VSS",  "category": "Intl Equity",     "description": "FTSE All-World ex-US Small Cap"},
    {"id": "em_equity",      "label": "Emerging Markets",       "etf": "EEM",  "category": "Intl Equity",     "description": "MSCI EM — China, India, Brazil, Taiwan"},
    # ── Real Assets ──
    {"id": "reits",          "label": "US Real Estate (REITs)", "etf": "VNQ",  "category": "Real Assets",     "description": "Vanguard Real Estate — income + inflation hedge"},
    {"id": "commodities",    "label": "Broad Commodities",      "etf": "PDBC", "category": "Real Assets",     "description": "Diversified commodity basket — energy, metals, ag"},
    {"id": "gold",           "label": "Gold",                   "etf": "GLD",  "category": "Real Assets",     "description": "SPDR Gold — crisis hedge, dollar hedge"},
    {"id": "energy",         "label": "Energy / Oil",           "etf": "XLE",  "category": "Real Assets",     "description": "S&P Energy Sector — oil majors, E&P"},
    # ── Fixed Income ──
    {"id": "bonds_st",       "label": "US Bonds Short-Term",    "etf": "SHY",  "category": "Fixed Income",    "description": "1-3yr Treasuries — capital preservation, low duration"},
    {"id": "bonds_it",       "label": "US Bonds Intermediate",  "etf": "IEF",  "category": "Fixed Income",    "description": "7-10yr Treasuries — moderate duration risk"},
    {"id": "bonds_lt",       "label": "US Bonds Long-Term",     "etf": "TLT",  "category": "Fixed Income",    "description": "20+yr Treasuries — high duration, bear market hedge"},
    {"id": "tips",           "label": "TIPS / Inflation-Linked","etf": "TIP",  "category": "Fixed Income",    "description": "Inflation-protected Treasuries — real return floor"},
    {"id": "high_yield",     "label": "High Yield Bonds",       "etf": "HYG",  "category": "Fixed Income",    "description": "US corporate junk bonds — equity-like credit risk"},
    {"id": "intl_bonds",     "label": "International Bonds",    "etf": "BNDX", "category": "Fixed Income",    "description": "Ex-US investment grade sovereign & corporate"},
    # ── Cash / Defensive ──
    {"id": "cash",           "label": "Cash / T-Bills",         "etf": "SGOV", "category": "Defensive",       "description": "0-3 month T-Bills — risk-free rate, bear market refuge"},
]

# Asset class IDs considered risk-on (equity/growth biased)
RISK_ON_ASSET_IDS = {
    "us_lc_growth", "us_lc_value", "us_sc_growth", "us_sc_value",
    "us_midcap", "us_momentum", "intl_dev", "intl_sc", "em_equity",
    "reits", "high_yield", "energy", "commodities",
}

# Asset class IDs considered risk-off (defensive/income)
RISK_OFF_ASSET_IDS = {
    "bonds_st", "bonds_it", "bonds_lt", "tips", "gold",
    "intl_bonds", "cash",
}


# ─────────────────────────────────────────────
#  Macro Scenario Framework
# ─────────────────────────────────────────────

MACRO_SCENARIOS: dict[str, dict] = {
    "bull": {
        "label": "Bull / Risk-On",
        "color": "#00ff88",
        "icon": "🐂",
        "description": (
            "Expansionary macro regime. Leading indicators improving, credit spreads tight, "
            "VIX suppressed (<16), yield curve steepening. Equities outperform, "
            "especially growth & momentum. Overweight risk assets; underweight bonds & cash."
        ),
        "regime_signals": [
            "VIX < 16 (fear absent)",
            "Yield curve steepening (10Y > 2Y)",
            "Credit spreads tight (HYG outperforming)",
            "Breadth expanding (small caps leading)",
            "ISM Manufacturing > 52",
        ],
        "active_asset_ids": {
            "us_lc_growth", "us_sc_growth", "us_midcap", "us_momentum",
            "em_equity", "intl_dev", "reits", "high_yield", "energy", "commodities",
        },
        "allocations": [
            {"asset_id": "us_lc_growth",  "weight": 30, "etf": "IWF",  "label": "US Large Cap Growth"},
            {"asset_id": "us_momentum",   "weight": 20, "etf": "MTUM", "label": "US Momentum"},
            {"asset_id": "us_sc_growth",  "weight": 15, "etf": "IJT",  "label": "US Small Cap Growth"},
            {"asset_id": "em_equity",     "weight": 15, "etf": "EEM",  "label": "Emerging Markets"},
            {"asset_id": "intl_dev",      "weight": 10, "etf": "EFA",  "label": "Intl Developed"},
            {"asset_id": "reits",         "weight":  5, "etf": "VNQ",  "label": "Real Estate"},
            {"asset_id": "high_yield",    "weight":  5, "etf": "HYG",  "label": "High Yield"},
        ],
    },
    "base": {
        "label": "Base / Balanced",
        "color": "#00d4ff",
        "icon": "⚖️",
        "description": (
            "Neutral macro regime. Mixed signals, moderate growth, "
            "VIX 16–24, relatively flat yield curve. Balanced equity/bond exposure. "
            "Focus on quality, reduce beta, maintain diversification across asset classes."
        ),
        "regime_signals": [
            "VIX 16–24 (moderate caution)",
            "Yield curve flat to modestly inverted",
            "Credit spreads modestly wider",
            "Mixed breadth — large cap leading small cap",
            "ISM Manufacturing 48–52 (near contraction/expansion boundary)",
        ],
        "active_asset_ids": {
            "us_lc_growth", "us_lc_value", "us_midcap", "intl_dev",
            "reits", "bonds_it", "tips", "gold",
        },
        "allocations": [
            {"asset_id": "us_lc_growth",  "weight": 25, "etf": "IWF",  "label": "US Large Cap Growth"},
            {"asset_id": "us_lc_value",   "weight": 15, "etf": "IWD",  "label": "US Large Cap Value"},
            {"asset_id": "intl_dev",      "weight": 15, "etf": "EFA",  "label": "Intl Developed"},
            {"asset_id": "bonds_it",      "weight": 20, "etf": "IEF",  "label": "Intermediate Bonds"},
            {"asset_id": "gold",          "weight": 10, "etf": "GLD",  "label": "Gold"},
            {"asset_id": "tips",          "weight": 10, "etf": "TIP",  "label": "TIPS"},
            {"asset_id": "reits",         "weight":  5, "etf": "VNQ",  "label": "Real Estate"},
        ],
    },
    "bear": {
        "label": "Bear / Risk-Off",
        "color": "#ff4d4d",
        "icon": "🐻",
        "description": (
            "Contractionary macro regime. Leading indicators deteriorating, credit spreads widening, "
            "VIX elevated (>24), yield curve inverted. Primary objective: AVOID LARGE LOSSES. "
            "Rotate to bonds, gold, TIPS, and cash. Minimize equity exposure. "
            "Cougar Global's core thesis: survive bear markets to compound through bull markets."
        ),
        "regime_signals": [
            "VIX > 24 (fear elevated)",
            "Yield curve deeply inverted (recession signal)",
            "Credit spreads widening sharply (HYG underperforming)",
            "Breadth collapsing (small caps lagging badly)",
            "ISM Manufacturing < 48 (contraction territory)",
        ],
        "active_asset_ids": {
            "bonds_lt", "bonds_it", "bonds_st", "tips", "gold", "cash", "intl_bonds",
        },
        "allocations": [
            {"asset_id": "bonds_lt",   "weight": 30, "etf": "TLT",  "label": "Long-Term Treasuries"},
            {"asset_id": "gold",       "weight": 25, "etf": "GLD",  "label": "Gold"},
            {"asset_id": "tips",       "weight": 20, "etf": "TIP",  "label": "TIPS"},
            {"asset_id": "bonds_it",   "weight": 15, "etf": "IEF",  "label": "Intermediate Treasuries"},
            {"asset_id": "cash",       "weight": 10, "etf": "SGOV", "label": "Cash / T-Bills"},
        ],
    },
}


# ─────────────────────────────────────────────
#  Macro Regime Detection (Heuristic)
# ─────────────────────────────────────────────

def assess_macro_regime(macro_data: list) -> dict:
    """
    Heuristically assess current macro regime from live indicators.
    Returns regime key ('bull'/'base'/'bear') + confidence + signal summary.
    """
    # Extract relevant indicators from macro_data list
    indicators = {item["label"]: item for item in macro_data}

    bull_signals = 0
    bear_signals = 0
    signal_log   = []

    # VIX
    vix_item = indicators.get("VIX")
    if vix_item:
        vix = vix_item["price"]
        if vix < 16:
            bull_signals += 2
            signal_log.append(f"VIX {vix:.1f} — fear absent (bullish)")
        elif vix < 24:
            signal_log.append(f"VIX {vix:.1f} — moderate caution (neutral)")
        else:
            bear_signals += 2
            signal_log.append(f"VIX {vix:.1f} — elevated fear (bearish)")

    # 10-yr yield trend (change)
    yield_item = indicators.get("10-Yr Yield")
    if yield_item:
        chg = yield_item.get("change", 0)
        lvl = yield_item["price"]
        if chg > 0.03 and lvl < 5.5:
            bull_signals += 1
            signal_log.append(f"10-Yr yield rising ({lvl:.2f}%) — reflation/growth (bullish)")
        elif lvl > 5.5:
            bear_signals += 1
            signal_log.append(f"10-Yr yield elevated ({lvl:.2f}%) — restrictive policy (bearish)")
        else:
            signal_log.append(f"10-Yr yield {lvl:.2f}% — neutral")

    # S&P 500 momentum (day change as proxy)
    sp_item = indicators.get("S&P 500")
    if sp_item:
        chg = sp_item.get("change", 0)
        if chg > 0.5:
            bull_signals += 1
            signal_log.append(f"S&P 500 +{chg:.2f}% today — positive momentum")
        elif chg < -1.0:
            bear_signals += 1
            signal_log.append(f"S&P 500 {chg:.2f}% today — negative momentum (bearish)")
        else:
            signal_log.append(f"S&P 500 {chg:.2f}% today — muted (neutral)")

    # Gold (flight to safety signal)
    gold_item = indicators.get("Gold")
    if gold_item:
        chg = gold_item.get("change", 0)
        if chg > 0.5:
            bear_signals += 1
            signal_log.append(f"Gold +{chg:.2f}% — flight-to-safety demand (bearish signal)")
        elif chg < -0.5:
            bull_signals += 1
            signal_log.append(f"Gold {chg:.2f}% — risk-on (bullish signal)")

    # NASDAQ relative to broad market
    ndx_item = indicators.get("NASDAQ 100")
    sp_item2  = indicators.get("S&P 500")
    if ndx_item and sp_item2:
        ndx_chg = ndx_item.get("change", 0)
        sp_chg  = sp_item2.get("change", 0)
        if ndx_chg > sp_chg + 0.3:
            bull_signals += 1
            signal_log.append(f"NASDAQ leading S&P by {ndx_chg-sp_chg:.2f}% — growth outperforming (bullish)")
        elif ndx_chg < sp_chg - 0.3:
            bear_signals += 1
            signal_log.append(f"NASDAQ lagging S&P by {sp_chg-ndx_chg:.2f}% — growth underperforming (bearish)")

    # Regime verdict
    if bull_signals == 0 and bear_signals == 0:
        # No live data: default to base
        regime = "base"
        confidence = 40
        signal_log.append("Insufficient live data — defaulting to Base regime")
    elif bull_signals > bear_signals + 1:
        regime = "bull"
        confidence = min(95, 50 + (bull_signals - bear_signals) * 12)
    elif bear_signals > bull_signals + 1:
        regime = "bear"
        confidence = min(95, 50 + (bear_signals - bull_signals) * 12)
    else:
        regime = "base"
        confidence = 55

    return {
        "regime":      regime,
        "confidence":  confidence,
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "signal_log":  signal_log,
        "scenario":    MACRO_SCENARIOS[regime],
    }


# ─────────────────────────────────────────────
#  20-Asset-Class Grid: In / Out Status
# ─────────────────────────────────────────────

def get_asset_class_grid(regime: str) -> list[dict]:
    """
    Return the 20-asset-class grid with 'in' / 'out' / 'watch' status
    for the given macro regime.
    """
    active_ids = MACRO_SCENARIOS.get(regime, MACRO_SCENARIOS["base"])["active_asset_ids"]
    grid = []
    for ac in ASSET_CLASS_UNIVERSE:
        if ac["id"] in active_ids:
            status = "IN"
        elif ac["id"] in RISK_ON_ASSET_IDS and regime == "base":
            status = "WATCH"
        elif ac["id"] in RISK_OFF_ASSET_IDS and regime == "base":
            status = "WATCH"
        else:
            status = "OUT"
        grid.append({**ac, "status": status})
    return grid


# ─────────────────────────────────────────────
#  Tactical Rebalancing Signals
# ─────────────────────────────────────────────

def get_tactical_rebalancing_signals(
    portfolio_df: pd.DataFrame,
    regime: str,
    investable_cash: float = 0.0,
) -> list[dict]:
    """
    Compare current portfolio holdings to Cougar Global scenario allocations.
    Returns a list of actionable signals: rotate, add, reduce, or hold.
    """
    scenario = MACRO_SCENARIOS.get(regime, MACRO_SCENARIOS["base"])
    target_allocs = {a["etf"]: a["weight"] for a in scenario["allocations"]}
    active_ids    = scenario["active_asset_ids"]

    # Map current portfolio tickers to asset classes
    etf_map = {ac["etf"]: ac for ac in ASSET_CLASS_UNIVERSE}
    current_weights = portfolio_df.set_index("Ticker")["Weight%"].to_dict()

    signals = []

    # Check each target position
    for alloc in scenario["allocations"]:
        etf     = alloc["etf"]
        target  = alloc["weight"]
        current = current_weights.get(etf, 0.0)
        gap     = target - current

        if current == 0 and target > 0:
            signals.append({
                "action":  "ADD",
                "ticker":  etf,
                "label":   alloc["label"],
                "current": current,
                "target":  target,
                "gap":     gap,
                "urgency": "High" if target >= 20 else "Medium",
                "msg":     f"Not in portfolio. Target {target}% in {regime} regime.",
            })
        elif gap > 5:
            signals.append({
                "action":  "INCREASE",
                "ticker":  etf,
                "label":   alloc["label"],
                "current": current,
                "target":  target,
                "gap":     gap,
                "urgency": "Medium",
                "msg":     f"Underweight by {gap:.1f}%. Increase from {current:.1f}% → {target}%.",
            })
        elif gap < -5:
            signals.append({
                "action":  "REDUCE",
                "ticker":  etf,
                "label":   alloc["label"],
                "current": current,
                "target":  target,
                "gap":     gap,
                "urgency": "Medium",
                "msg":     f"Overweight by {abs(gap):.1f}%. Trim from {current:.1f}% → {target}%.",
            })
        else:
            signals.append({
                "action":  "HOLD",
                "ticker":  etf,
                "label":   alloc["label"],
                "current": current,
                "target":  target,
                "gap":     gap,
                "urgency": "Low",
                "msg":     f"On target. Current {current:.1f}% ≈ target {target}%.",
            })

    # Flag positions that should be OUT in this regime
    out_ids    = set(ac["id"] for ac in ASSET_CLASS_UNIVERSE) - active_ids
    out_etfs   = {ac["etf"] for ac in ASSET_CLASS_UNIVERSE if ac["id"] in out_ids}
    for ticker, weight in current_weights.items():
        if ticker in out_etfs and weight > 3:
            ac_info = etf_map.get(ticker, {})
            signals.append({
                "action":  "EXIT",
                "ticker":  ticker,
                "label":   ac_info.get("label", ticker),
                "current": weight,
                "target":  0,
                "gap":     -weight,
                "urgency": "High" if weight > 10 else "Medium",
                "msg":     f"Asset class out of model for {regime} regime. Consider rotating proceeds to target positions.",
            })

    # Sort: EXIT/ADD first, then by urgency
    priority = {"EXIT": 0, "ADD": 1, "INCREASE": 2, "REDUCE": 3, "HOLD": 4}
    signals.sort(key=lambda x: (priority.get(x["action"], 5), -abs(x["gap"])))
    return signals


# ─────────────────────────────────────────────
#  Scenario-specific allocation plan
# ─────────────────────────────────────────────

def generate_scenario_allocation_plan(cash: float, regime: str) -> list[dict]:
    """
    Generate a deployment plan for investable cash based on the active macro regime,
    following Cougar Global's tactical allocation methodology.
    """
    if cash <= 0:
        return []
    scenario = MACRO_SCENARIOS.get(regime, MACRO_SCENARIOS["base"])
    plan = []
    for i, alloc in enumerate(scenario["allocations"]):
        ac = next((a for a in ASSET_CLASS_UNIVERSE if a["id"] == alloc["asset_id"]), {})
        plan.append({
            "rank":       i + 1,
            "ticker":     alloc["etf"],
            "name":       alloc["label"],
            "alloc_pct":  alloc["weight"],
            "amount":     round(cash * alloc["weight"] / 100, 2),
            "rationale":  ac.get("description", ""),
            "priority":   "Buy First" if i < 2 else ("Buy Next" if i < 4 else "Optional"),
            "regime":     regime,
        })
    return plan
