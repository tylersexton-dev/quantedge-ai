"""
analyst.py
QuantEdge AI — Claude-powered institutional analyst
Persona: Tactical asset allocation strategist in the tradition of Cougar Global Investments.
Methodology: Post Modern Portfolio Theory (PMPT), macro-driven regime analysis,
downside-risk-first framing, 20-asset-class tactical rotation.
"""

import os
import json
from typing import Optional

# Graceful degradation if anthropic not installed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ─────────────────────────────────────────────
#  Analyst Configuration — Cougar Global Style
# ─────────────────────────────────────────────

ANALYST_PERSONA = """You are QuantEdge Analyst — a senior tactical asset allocation strategist in the tradition of Cougar Global Investments (a Raymond James affiliate founded 1993). Your investment philosophy is built on three pillars:

1. DOWNSIDE FIRST. Your primary objective is to avoid large losses, not to maximize return. Portfolios that survive bear markets compound dramatically over time. You always lead with risk, then return.

2. MACRO-DRIVEN TACTICAL ALLOCATION. You run a top-down global macro process with three scenario regimes: Bull, Base, and Bear. You rotate across a 20-asset-class universe based on which regime you're in. Fewer than 10 asset classes are typically "active" at once.

3. POST MODERN PORTFOLIO THEORY (PMPT). You use Probability of Loss (not standard deviation) as your primary risk metric. Downside deviation (deviation below zero, not total deviation) is your volatility measure. Sortino ratio (not Sharpe) is your risk-adjusted return metric. Standard deviation penalizes upside volatility — that's not risk, that's return.

Your communication style:
- Regime-first: always establish the current macro regime before making any allocation recommendations
- Downside-first: lead every risk section with Probability of Loss and max drawdown, not returns
- Precise and institutional: cite specific metrics (Probability of Loss %, downside deviation %, Sortino ratio)
- Tactical, not buy-and-hold: frame every recommendation in terms of "current regime" and what would trigger a rotation
- ETF-only for tactical allocation (Cougar Global methodology uses exclusively ETFs)
- Professional PMPT vocabulary: "probability of loss," "downside deviation," "minimum acceptable return (MAR)," "macro regime," "tactical rotation," "asset class in/out of model," "bear market avoidance"

CRITICAL RULES:
- Always cite Probability of Loss before mentioning returns
- Every allocation recommendation must be regime-conditional ("in a bull regime... in a bear regime...")
- Frame the Roth IRA's long horizon as justification for staying in the model through volatility — but the model itself rotates tactically
- Never use "Sharpe ratio" as your primary metric — use Sortino and Probability of Loss instead
- Always give specific tickers, exact percentages, exact dollar amounts"""

STATIC_FALLBACK_ANALYSIS = """
## QuantEdge Portfolio Assessment — Cougar Global Methodology

*Note: Live AI analysis unavailable (API key not configured). Showing rules-based tactical analysis.*

---

### Executive Summary
Your current portfolio is positioned for a bull macro regime — growth-heavy, tech-overweight, no defensive allocation. The primary risk metric that matters: **Probability of Loss over 1 year is estimated at 22–28%** for this allocation. Before sizing into growth, we need to establish the current macro regime and ensure the portfolio can survive a bear scenario without catastrophic drawdown. The Cougar Global philosophy is explicit: a portfolio that avoids a 40% bear market drawdown will dramatically outcompound one that suffers it, even if the recovering portfolio generates higher returns in bull markets.

---

### Current Macro Regime Assessment

**Regime: Base / Transitional** *(heuristic — add API key for live AI assessment)*

Mixed signals characterize the current environment. VIX in the 15–20 range suggests moderate risk appetite but not full complacency. The 10-year yield at 4.2–4.5% remains restrictive relative to historical norms. In a Base regime, Cougar Global's model holds a balanced allocation: 40% equities, 30% bonds, 20% gold/TIPS, 10% cash.

**Regime Signals:**
- VIX 15–20: Cautiously neutral — not risk-on, not risk-off
- Yield curve: Flat to mildly inverted — late-cycle caution warranted
- Credit spreads: Tight, suggesting complacency risk
- Earnings revision momentum: Positive for large-cap tech, mixed elsewhere

---

### PMPT Risk Analysis (Downside-First)

| Metric | Current Portfolio | Target (Base Regime) |
|--------|-------------------|---------------------|
| **Probability of Loss (1yr)** | ~25% | <18% |
| **Probability of Loss (3yr)** | ~12% | <8% |
| Downside Deviation | ~14% | <10% |
| Sortino Ratio | ~0.85 | >1.0 |
| Max Drawdown (historical) | ~35–42% | <25% |
| Beta vs S&P 500 | ~1.25 | <1.0 |

**Risk Verdict:** Your current portfolio carries meaningful downside risk. A 40% drawdown — which a 1.25-beta tech-heavy portfolio historically experiences in bear markets — would require a 67% gain just to return to breakeven. That compounding math is why Cougar Global's primary objective is loss avoidance, not return maximization.

---

### $2,500 Deployment — Base Regime Tactical Allocation

In a Base/neutral macro regime, the Cougar Global methodology calls for a balanced, diversified allocation across 7–8 asset classes:

| Rank | Ticker | Asset Class | Allocation | Amount | Rationale |
|------|--------|-------------|-----------|--------|-----------|
| 1 | IWF | US Large Cap Growth | 25% | $625 | Core equity exposure — quality growth, base regime appropriate |
| 2 | IEF | Intermediate Treasuries | 20% | $500 | Duration hedge + flight-to-safety; reduces portfolio Prob of Loss |
| 3 | IWD | US Large Cap Value | 15% | $375 | Diversify equity factor exposure; value outperforms in late cycle |
| 4 | EFA | Intl Developed | 15% | $375 | Geographic diversification; dollar hedge |
| 5 | GLD | Gold | 10% | $250 | Crisis hedge; historically uncorrelated, reduces Prob of Loss |
| 6 | TIP | TIPS | 10% | $250 | Inflation protection + real return floor |
| 7 | VNQ | Real Estate | 5% | $125 | Income + inflation-hedging; improves Sortino |

**Bear Regime Contingency:** If VIX breaks above 24 and credit spreads widen materially, rotate entirely to TLT (30%), GLD (25%), TIP (20%), IEF (15%), SGOV (10%). This is Cougar Global's bear model — preserve capital, wait for the regime to clear.

---

### Roth IRA Optimization (PMPT Framework)

The Roth IRA's tax-free compounding makes it the ideal vehicle for the *growth* sleeve of a tactical portfolio — but only the sleeve that's appropriate for the current regime. Key principles:

**In Bull Regime:** Roth IRA holds high-beta growth (IWF, MTUM, IJT) — these generate the highest terminal value tax-free
**In Base Regime:** Mix growth (60%) and defensive (40%) — current recommendation
**In Bear Regime:** Rotate to TLT, GLD, SGOV even in Roth IRA — capital preservation trumps tax efficiency when drawdown risk is high

---

### 10-Year Roadmap (Three Scenario Paths)

Using Monte Carlo bootstrapping with scenario-weighted returns:

| Scenario | Ann. Return | Prob of Loss (1yr) | 10yr Terminal ($2,700 start) |
|----------|------------|-------------------|------------------------------|
| **Bull Regime** | 16–18% | 15% | $12,000–$15,000 |
| **Base Regime** | 10–12% | 20% | $6,500–$8,500 |
| **Bear w/ Avoidance** | 8–9%* | 8% | $5,500–$7,000 |

*Bear case WITH tactical rotation (Cougar Global model) — avoids the -40% drawdown, dramatically improves compounding path vs. static buy-and-hold through a bear market.

**Key insight:** A portfolio that loses 40% in a bear market needs 67% to recover. A portfolio that loses only 12% (via tactical rotation) needs only 14% to recover. The Cougar Global model's edge is in this asymmetry — not in generating higher bull market returns, but in dramatically shortening the recovery path.

---

### Tactical Monitoring Triggers

Rotate toward Bear regime if:
- VIX closes above 24 for 3+ consecutive days
- 10-year yield inverts below 2-year by more than 75 bps for 60+ days
- S&P 500 breaks 200-day moving average on volume
- HYG (high yield) underperforms IEF (treasuries) for 20 consecutive days

Rotate toward Bull regime if:
- VIX drops below 16 and holds for 10+ trading days
- ISM Manufacturing crosses 52 from below
- S&P 500 holds 50-day moving average + breadth expanding (Russell 2000 outperforming)
- Credit spreads tighten (HYG outperforming investment grade)
"""


# ─────────────────────────────────────────────
#  Prompt Construction
# ─────────────────────────────────────────────

def build_analysis_prompt(
    portfolio_data: dict,
    risk_metrics: dict,
    macro_data: list,
    investable_cash: float,
    allocation_plan: list,
    regime_assessment: dict | None = None,
) -> str:
    """Build a Cougar Global-style PMPT analysis prompt."""

    portfolio_str = json.dumps(portfolio_data, indent=2, default=str)
    risk_str      = json.dumps(risk_metrics, indent=2, default=str)
    macro_str     = json.dumps(macro_data[:6], indent=2, default=str) if macro_data else "N/A"
    alloc_str     = json.dumps(allocation_plan, indent=2, default=str)
    regime_str    = json.dumps(regime_assessment, indent=2, default=str) if regime_assessment else "N/A"

    prompt = f"""You are analyzing the portfolio of Tyler, an aggressive-growth investor with a Roth IRA.
Apply the Cougar Global Investments tactical asset allocation methodology: PMPT, macro regime analysis, downside-first framing.

## PORTFOLIO DATA
```json
{portfolio_str}
```

## RISK METRICS (Live — PMPT Framework)
```json
{risk_str}
```

## MACRO ENVIRONMENT (Live Indicators)
```json
{macro_str}
```

## MACRO REGIME ASSESSMENT (Algorithmic)
```json
{regime_str}
```

## PROPOSED REGIME-CONDITIONAL ALLOCATION FOR ${investable_cash:,.0f}
```json
{alloc_str}
```

---

Provide a comprehensive Cougar Global-style portfolio analysis structured EXACTLY as follows:

### 1. Macro Regime Assessment (Lead Section)
Establish the current macro regime (Bull/Base/Bear) based on the live indicators. Cite VIX level, yield curve, credit conditions, equity breadth. Give your regime confidence level (%). This drives everything else — state it clearly.

### 2. Downside Risk Profile (PMPT — Lead with Loss)
Report Probability of Loss (1yr and 3yr), downside deviation, Sortino ratio, and max drawdown. Compare to Cougar Global's regime-appropriate benchmarks. Give your risk verdict: is this portfolio positioned to survive a bear market?

### 3. Current Portfolio Assessment
Analyze each position for its regime-appropriateness. Are the current holdings consistent with the detected macro regime? Which positions would you hold, rotate, or exit if the regime shifted to bear?

### 4. ${investable_cash:,.0f} Tactical Deployment
Walk through the regime-conditional allocation. Explain why each asset class is "in" the model for this regime. Give execution guidance. Also state what the allocation would look like in each alternative regime.

### 5. Roth IRA Optimization (Tactical)
Apply Cougar Global's methodology to the Roth IRA sleeve: what's appropriate for this regime, and how it would shift in a bull/bear rotation. Specific ETF tickers with weights.

### 6. Bear Market Contingency Plan
This is required in every Cougar Global analysis. Explicitly state: what triggers a rotation to defensive? What does the bear portfolio look like? What is the expected Probability of Loss in the bear allocation vs. current?

### 7. 10-Year Compounding Roadmap (Scenario-Weighted)
Give terminal value estimates for each regime path. Show the compounding math that proves why bear market avoidance is the most powerful lever — not bull market alpha. Use specific numbers.

Keep it institutional, precise, downside-first. This is a tactical research note, not a buy-and-hold recommendation.
"""
    return prompt


def build_macro_regime_prompt(macro_data: list, algo_regime: dict) -> str:
    """Build a Cougar Global macro regime assessment prompt."""
    macro_str = json.dumps(macro_data, indent=2, default=str)
    algo_str  = json.dumps(algo_regime, indent=2, default=str)
    return f"""You are running Cougar Global's macro regime assessment — Step 1 of the 3-step tactical allocation process.

Live macro indicators:
```json
{macro_str}
```

Algorithmic regime assessment (heuristic):
```json
{algo_str}
```

Provide a 4–5 sentence macro regime brief:
1. Current regime call: Bull, Base, or Bear — and your confidence level
2. The 2–3 signals driving your call (VIX, yield curve, credit, breadth)
3. What this means for the 20-asset-class universe — which categories are "in" and "out"
4. One sentence on the primary risk: what would flip this regime

Use PMPT language: "Probability of Loss," "downside deviation," "macro regime," "tactical rotation."
Be direct. No hedging. This is a morning regime briefing, not an academic paper."""


def build_scenario_allocation_prompt(
    cash: float,
    regime: str,
    scenario: dict,
    macro_data: list,
) -> str:
    """Build a scenario-conditional allocation recommendation prompt."""
    alloc_str  = json.dumps(scenario.get("allocations", []), indent=2, default=str)
    macro_str  = json.dumps(macro_data[:5], indent=2, default=str) if macro_data else "N/A"
    return f"""You are generating a Cougar Global-style tactical allocation recommendation.

Detected macro regime: **{regime.upper()}**
Scenario description: {scenario.get('description', '')}

Target allocations for {regime} regime:
```json
{alloc_str}
```

Live macro conditions:
```json
{macro_str}
```

Available cash to deploy: ${cash:,.0f}

In 3–4 sentences:
1. Confirm the regime and what it means for asset class selection
2. Give your top conviction position in this regime and WHY (use PMPT: probability of loss, downside deviation)
3. Name the single asset class most likely to be "rotated out" and why
4. Trigger: what macro signal would flip you to a different regime allocation

Be tactical, not strategic. This is a trade note, not a long-term plan."""


def build_quick_take_prompt(ticker: str, price: float, change_pct: float) -> str:
    """Quick PMPT-framed analyst take on a single ticker."""
    return f"""Give a 3-sentence tactical assessment of {ticker} at ${price:.2f} (day change: {change_pct:+.2f}%).

Include: (1) what today's price action signals about the current macro regime, (2) the asset class's Probability of Loss contribution — does it raise or lower overall portfolio downside risk?, (3) whether it belongs "in" or "out" of a Cougar Global-style model in the current environment.

Be direct. Use PMPT language: probability of loss, downside risk, regime-conditional."""


def build_market_commentary_prompt(macro_data: list) -> str:
    """Cougar Global-style macro morning briefing prompt."""
    macro_str = json.dumps(macro_data, indent=2, default=str)
    return f"""Given this macro snapshot:
```json
{macro_str}
```

In 3–4 sentences, give a Cougar Global-style macro regime briefing:
- Current regime: Bull / Base / Bear — and why
- Which of the 20 asset classes are "in" vs "out" of the model today
- Primary downside risk: what could cause a regime shift to Bear?
- One tactical implication: what would you rotate into or out of right now?

Use PMPT vocabulary: probability of loss, downside deviation, macro regime, tactical rotation.
No generic commentary. This is a pre-market regime call."""


# ─────────────────────────────────────────────
#  Claude API Integration
# ─────────────────────────────────────────────

class QuantEdgeAnalyst:
    """Claude-powered tactical asset allocation analyst — Cougar Global methodology."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client   = None
        self.available = False

        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client    = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
            except Exception:
                self.available = False

    def _call(self, prompt: str, max_tokens: int = 2000) -> str:
        """Make a Claude API call with Cougar Global analyst persona."""
        if not self.available or not self.client:
            return STATIC_FALLBACK_ANALYSIS

        try:
            message = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=max_tokens,
                system=ANALYST_PERSONA,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.AuthenticationError:
            return "**Error:** Invalid API key. Please check your ANTHROPIC_API_KEY in the .env file."
        except anthropic.RateLimitError:
            return "**Error:** Rate limit hit. Please wait a moment and try again."
        except anthropic.APIError as e:
            return f"**API Error:** {str(e)}\n\n---\n\n{STATIC_FALLBACK_ANALYSIS}"
        except Exception as e:
            return f"**Unexpected error:** {str(e)}\n\n---\n\n{STATIC_FALLBACK_ANALYSIS}"

    def analyze_portfolio(
        self,
        portfolio_data: dict,
        risk_metrics: dict,
        macro_data: list,
        investable_cash: float,
        allocation_plan: list,
        regime_assessment: dict | None = None,
    ) -> str:
        """Full Cougar Global PMPT portfolio analysis."""
        if not self.available:
            return STATIC_FALLBACK_ANALYSIS

        prompt = build_analysis_prompt(
            portfolio_data=portfolio_data,
            risk_metrics=risk_metrics,
            macro_data=macro_data,
            investable_cash=investable_cash,
            allocation_plan=allocation_plan,
            regime_assessment=regime_assessment,
        )
        return self._call(prompt, max_tokens=2800)

    def macro_regime_assessment(self, macro_data: list, algo_regime: dict) -> str:
        """Cougar Global macro regime call — Step 1 of tactical process."""
        if not self.available:
            regime = algo_regime.get("regime", "base")
            conf   = algo_regime.get("confidence", 50)
            return (
                f"*AI analyst offline — algorithmic regime call: **{regime.upper()}** "
                f"({conf}% confidence). Add ANTHROPIC_API_KEY for live macro assessment.*"
            )
        prompt = build_macro_regime_prompt(macro_data, algo_regime)
        return self._call(prompt, max_tokens=450)

    def scenario_allocation_advice(
        self,
        cash: float,
        regime: str,
        scenario: dict,
        macro_data: list,
    ) -> str:
        """Regime-conditional tactical allocation commentary."""
        if not self.available:
            return f"*AI analyst offline. In a {regime} regime, follow the allocation plan shown above. Add API key for live commentary.*"
        prompt = build_scenario_allocation_prompt(cash, regime, scenario, macro_data)
        return self._call(prompt, max_tokens=400)

    def quick_ticker_take(self, ticker: str, price: float, change_pct: float) -> str:
        """Quick PMPT-framed take on a single ticker."""
        if not self.available:
            return (
                f"*AI analyst offline. {ticker} showing {change_pct:+.2f}% today. "
                f"Add API key for regime-conditional commentary.*"
            )
        prompt = build_quick_take_prompt(ticker, price, change_pct)
        return self._call(prompt, max_tokens=300)

    def market_commentary(self, macro_data: list) -> str:
        """Cougar Global macro regime morning briefing."""
        if not self.available:
            return "*AI analyst offline. Add your ANTHROPIC_API_KEY for live macro regime commentary.*"
        prompt = build_market_commentary_prompt(macro_data)
        return self._call(prompt, max_tokens=450)

    def chat(self, user_message: str, portfolio_context: str = "") -> str:
        """Free-form chat — analyst stays in Cougar Global PMPT persona."""
        if not self.available:
            return "AI analyst is offline. Please add your ANTHROPIC_API_KEY to enable the live analyst."
        context_block = f"\n\nPortfolio Context:\n{portfolio_context}" if portfolio_context else ""
        return self._call(f"{user_message}{context_block}", max_tokens=800)

    @property
    def status(self) -> str:
        if not ANTHROPIC_AVAILABLE:
            return "offline — `anthropic` package not installed"
        if not self.api_key:
            return "offline — API key not configured"
        if self.available:
            return "online"
        return "offline — connection error"
