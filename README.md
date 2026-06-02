# ⚡ QuantEdge AI

<!-- ── Badges ─────────────────────────────────────────────────────────────── -->
<!-- Replace the Streamlit URL below with your deployed app URL after launch  -->
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Claude AI](https://img.shields.io/badge/Powered%20by-Claude%20AI-purple?logo=anthropic)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-powered tactical portfolio analyst.** Institutional-grade intelligence built on Cougar Global Investments' macro-driven methodology — Probability of Loss as the primary risk metric, 20-asset-class tactical rotation, and Claude AI for hedge-fund-grade commentary.

---

## 🚀 One-Click Deploy to Streamlit Cloud

**Step 1** — Fork or push this repo to GitHub (run `./deploy.sh` to automate)

**Step 2** — Go to [share.streamlit.io](https://share.streamlit.io) → New app → point at `app.py`

**Step 3** — In *Advanced settings → Secrets*, add:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Step 4** — Click **Deploy**. Done in ~2 minutes.

---

## What It Does

QuantEdge AI is a full-stack investment analytics platform built around three pillars from **Cougar Global Investments** (a Raymond James affiliate):

| Pillar | Description |
|--------|-------------|
| **Downside First** | Probability of Loss is the lead metric — not Sharpe ratio, not raw returns |
| **Macro-Driven TAA** | Top-down regime detection (Bull/Base/Bear) drives all allocation decisions |
| **Post Modern Portfolio Theory** | Downside deviation, Sortino ratio, bootstrapped POL simulation |

---

## Features

### 📊 Dashboard
Real-time portfolio valuation via yfinance. **Probability of Loss (1yr & 3yr)** displayed prominently as the primary risk metric alongside bear floor/bull ceiling. Sector breakdown, holdings table, and regime-conditional deployment plan.

### 🎯 Macro Framework *(new — Cougar Global core)*
Live macro regime detection (Bull / Base / Bear) from VIX, yield curve, S&P momentum, and gold. Three-scenario allocation comparison with animated bars. 20-asset-class IN/WATCH/OUT grid. Tactical rebalancing signals (ADD/EXIT/INCREASE/REDUCE) based on current regime vs. holdings.

### 🤖 AI Analyst
Claude-powered analyst in the tradition of Cougar Global — speaks in PMPT language: "probability of loss," "downside deviation," "macro regime," "tactical rotation." Full portfolio analysis: regime assessment → risk profile → deployment plan → bear contingency → 10-year roadmap. Graceful offline fallback (rules-based static analysis) when API key is absent.

### 📈 Roadmap
500-path Monte Carlo simulation. **Probability of Loss by year** charted as a bar chart (the Cougar Global primary output). Bear floor/bull ceiling shown with confidence bands. Year-by-year milestone table. Roth IRA tax-free advantage calculator.

### 🌍 Market Intel
8 live macro indicators with regime implication labels. Portfolio universe mover chart. Factor ETF dashboard (MTUM, QUAL, USMV, XMMO, IWF, VTV) with regime-match highlights.

### 💬 Chat
Free-form conversation with the AI analyst — stays in PMPT/Cougar Global persona throughout. Pre-loaded regime-aware quick prompts.

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/quantedge-ai.git
cd quantedge-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (optional — app works without it)
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-your-key-here

# 4. Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
quantedge-ai/
│
├── app.py                        # Main Streamlit application (1 150+ lines)
├── portfolio_engine.py           # Market data, PMPT metrics, Cougar Global framework
├── analyst.py                    # Claude API integration & analyst persona
│
├── .streamlit/
│   ├── config.toml               # Dark theme + server config
│   └── secrets.toml.example      # Template for Streamlit Cloud secrets
│
├── requirements.txt              # Python dependencies
├── packages.txt                  # System apt packages (intentionally minimal)
├── .gitignore                    # Excludes .env and secrets
├── .env.example                  # Local API key template
├── deploy.sh                     # One-command GitHub + Streamlit Cloud setup
└── README.md                     # This file
```

---

## Streamlit Cloud Secrets

The app reads `ANTHROPIC_API_KEY` from (in order):
1. **Streamlit Cloud secrets** — `st.secrets["ANTHROPIC_API_KEY"]`
2. **Local `.env` file** — loaded via `python-dotenv`
3. **Shell environment** — `os.environ["ANTHROPIC_API_KEY"]`
4. **Sidebar input** — paste directly in the UI at runtime

To configure on Streamlit Cloud: App → Settings → Secrets → paste:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

> **No API key?** The app still runs fully. The AI analyst tabs fall back to a high-quality rules-based static analysis using Cougar Global methodology. All charts, risk metrics, and the 20-asset-class framework work without any API key.

---

## Risk Metrics Explained (PMPT Framework)

| Metric | PMPT Role | Cougar Global View |
|--------|-----------|-------------------|
| **Probability of Loss** | Primary risk metric | "The only risk that matters is losing money" |
| **Downside Deviation** | PMPT volatility measure | Penalises downside only — not upside volatility |
| **Sortino Ratio** | Risk-adjusted return | Preferred over Sharpe — uses downside deviation |
| **Max Drawdown** | Bear market survival | Key to long-term compounding: avoid the -40% |
| Sharpe Ratio | Secondary / reference | Shown for context; not the primary lens |

---

## Macro Regime Framework

```
                   ┌─────────────────────────────────────────────┐
                   │        Cougar Global 3-Step Process          │
                   └─────────────────────────────────────────────┘
                            │
          ┌─────────────────┼────────────────────┐
          ▼                 ▼                    ▼
    🐂 BULL REGIME    ⚖️ BASE REGIME      🐻 BEAR REGIME
    VIX < 16          VIX 16–24           VIX > 24
    10 asset classes  8 asset classes     7 asset classes
    active            active              active
    
    IWF 30%           IWF 25%             TLT 30%
    MTUM 20%          IEF 20%             GLD 25%
    IJT 15%           IWD 15%             TIP 20%
    EEM 15%           EFA 15%             IEF 15%
    EFA 10%           GLD 10%             SGOV 10%
    VNQ 5%            TIP 10%
    HYG 5%            VNQ 5%
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit 1.35+ |
| Theme | Custom Bloomberg dark (#0a0e1a, #00d4ff, #ffd700) |
| Market Data | yfinance (curl_cffi backend) |
| AI Analyst | Anthropic Claude claude-opus-4-5 |
| Charts | Plotly 5.x (interactive) |
| Risk Engine | NumPy/Pandas (PMPT metrics, Monte Carlo) |
| Secrets | Streamlit secrets + python-dotenv |

---

## Disclaimer

QuantEdge AI is for **educational and informational purposes only**. It is not financial advice. All investments carry risk including loss of principal. Past performance does not guarantee future results. The Cougar Global methodology presented here is an interpretation for educational purposes — not an official product of Cougar Global Investments or Raymond James.

Always consult a qualified financial advisor before investing.

---

## License

MIT — use freely, modify freely, deploy and sell freely.

---

*Built with Claude AI · Powered by yfinance · Styled for Bloomberg terminals · Methodology inspired by Cougar Global Investments*
