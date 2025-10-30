# Investment Memo: FX Carry Strategy Analysis

**To:** Investment Committee  
**From:** Case Study Analysis  
**Date:** January 2025  
**Re:** FX Carry Strategy Evaluation (2000-2025)

---

## Executive Summary

Our analysis of a long-short FX carry strategy over 25 years (2000-2025) reveals that **carry is primarily a risk premium, not a free lunch**. The strategy generated negative returns with a -2.5% CAGR and -0.23 Sharpe ratio, suffering a catastrophic -67% maximum drawdown. While carry can provide diversification (near-zero equity correlation), it exhibits severe crash risk during market stress periods, making it unsuitable as a standalone strategy. **We do not recommend implementation without significant risk management overlays.**

## Methodology

**Data Analyzed:**
- **Period:** January 2000 - January 2025 (25 years, 9,100+ daily observations)
- **Currencies:** 8 developed and emerging markets (AUD, BRL, CAD, CHF, EUR, GBP, JPY, MXN)
- **Strategy:** Monthly rebalanced long-short portfolio (long top 3 highest carry, short bottom 3 lowest carry, equal-weighted at 1/3 each)

**Approach:**
Calculated excess returns combining FX appreciation and interest rate differentials, then analyzed risk-return profiles, drawdowns, and correlations with equity markets.

## Key Findings

### 1. Carry is Risk Compensation, Not a Free Lunch

**The evidence is clear:** Higher-carry currencies don't deliver free profits—they compensate investors for taking on significant risks:

- **Negative skewness:** High-carry currencies (BRL, MXN, AUD) show severe left-tail risk with skewness around -0.7 to -0.8, meaning devastating losses are more common than large gains
- **Extreme kurtosis:** Values of 12-43 indicate "fat tails"—extreme moves happen far more often than normal distributions would predict
- **Crash behavior:** Losses cluster during crises (2008 financial crisis, 2020 COVID crash, EM currency crises)
- **Safe haven pattern:** Low-carry currencies (CHF, JPY) appreciate during stress, acting as hedges

**Bottom line:** You're not outsmarting the market—you're being paid to provide liquidity and bear crash risk.

### 2. Strategy Performance: Disappointing Results

**Performance Metrics (2000-2025):**
- **CAGR:** -2.52% (negative returns over 25 years)
- **Sharpe Ratio:** -0.23 (poor risk-adjusted performance)
- **Maximum Drawdown:** -66.74% (catastrophic losses)
- **Volatility:** 9.20% annualized (moderate)
- **Correlation to Equities:** -0.04 (essentially zero—good for diversification)

**What this means:**
- $100 invested in 2000 would be worth only $40 by 2025
- The strategy lost money in 2 out of every 3 market regimes
- During the worst period, investors lost two-thirds of their capital
- The only bright spot: near-zero correlation to stocks provides some diversification value

### 3. Currency-Equity Relationships: Pro-Cyclical Risk

**Critical finding:** Higher-carry currencies behave badly when you need them most.

- **High-carry currencies** (BRL, MXN) show **slight negative correlation** to equities during normal times but **crash together during crises**
- **Low-carry currencies** (JPY, CHF) act as **safe havens**, appreciating when stocks fall
- This creates **adverse tail correlation**—the strategy suffers exactly when portfolios need protection

**Volatility regime analysis:**
- **High volatility periods:** -1.27 bps/day returns, Sharpe -0.29 (terrible)
- **Low volatility periods:** -0.42 bps/day returns, Sharpe -0.15 (still negative)
- Strategy performs poorly in BOTH environments, but especially during market stress

### 4. Risk Characteristics: The "Steamroller" Problem

This is a classic "picking up pennies in front of a steamroller" strategy:

**Individual Currency Performance:**
- **BRL (Brazil):** Highest carry (+10 bps) but worst returns (-1.3 bps/day), 90% volatility, fat tails
- **MXN (Mexico):** High carry (+5 bps) but negative returns (-0.9 bps/day), extreme skewness -0.8
- **CHF (Switzerland):** Low carry (-1.4 bps) but positive returns (+0.6 bps/day), best Sharpe 0.18

**The pattern:** Currencies offering high interest rates did so for good reason—they were risky and ultimately underperformed.

## Investment View

### Strengths:
- ✓ **Diversification:** Near-zero correlation to equities provides portfolio diversification benefits
- ✓ **Liquidity:** Currency markets are deep and liquid, allowing efficient implementation
- ✓ **Economic rationale:** Clear theoretical basis (uncovered interest parity violation)
- ✓ **Transparency:** Simple, rules-based strategy easy to understand and monitor

### Critical Concerns:
- ✗ **Negative returns:** Lost 60% over 25 years—this is not a viable standalone strategy
- ✗ **Crash risk:** -67% drawdown is unacceptable for most institutional mandates
- ✗ **Pro-cyclical:** Performs worst during market stress when diversification is most valuable
- ✗ **Structural headwinds:** High-carry currencies (EM) fundamentally riskier than low-carry (safe havens)
- ✗ **No positive expected return:** Strategy shows negative returns in ALL analyzed volatility regimes

### Recommendation: **DO NOT IMPLEMENT** (as currently structured)

**Rationale:**
1. **Returns don't justify risks:** Negative Sharpe ratio means you're not compensated for volatility taken
2. **Diversification is the only benefit:** But there are better ways to diversify (commodities, real assets, truly defensive strategies)
3. **Requires extensive risk management:** Would need volatility targeting, drawdown controls, and complementary strategies—significantly increasing complexity and costs

**If carry is pursued, it must be:**
- Part of a **multi-strategy FX approach** (combine with trend-following, value signals)
- **Heavily risk-managed** (max 5-10% of alternatives allocation, strict stop-losses)
- **Actively managed** with dynamic position sizing based on volatility regimes
- **Combined with defensive strategies** that perform during crises

## Framework for Evaluating Predictive Signals

To improve carry strategies, macro signals can help time exposures:

**1. Volatility Regime Filter:**
- **Finding:** Strategy performs -0.85 bps/day worse in high-vol vs low-vol regimes
- **Application:** Reduce or eliminate carry exposure when equity volatility > historical median
- **Information Coefficient:** -0.33 (moderately predictive)

**2. Evaluation Framework for Other Signals:**
- **Credit spreads:** Widening spreads = reduce exposure (flight to safety coming)
- **Central bank policy:** Track rate differential changes proactively
- **Carry momentum:** Recent performance predicts near-term returns
- **Risk appetite indicators:** VIX, high-yield spreads, currency volatility

**3. Key Testing Principles:**
- Always validate **out-of-sample** (avoid overfitting)
- Measure **Information Coefficient** (correlation between signal and forward returns)
- Check **economic significance** (does signal edge exceed transaction costs?)
- Test **robustness** across time periods and currencies

**Bottom line:** Signals can help, but they don't fix the fundamental problem—this strategy has negative expected returns even after accounting for risk.

---

## Conclusion

FX carry is **NOT a free lunch**—it's compensation for bearing crash risk, liquidity risk, and pro-cyclical exposure. Over 25 years, this compensation proved insufficient, delivering negative returns with devastating drawdowns. While the strategy offers diversification, there are superior alternatives that don't require investors to accept -67% drawdowns. Unless combined with sophisticated risk management and complementary strategies, FX carry should be avoided.

---

## Appendix: Key Charts Referenced
1. Cumulative returns by currency (showing CHF outperforming high-carry currencies)
2. Strategy drawdown chart (highlighting -67% maximum loss)
3. Volatility regime comparison (demonstrating poor performance in both environments)
4. Currency correlation heatmap (showing diversification across currencies)
5. Carry vs equity correlation scatter (revealing pro-cyclical tendencies)
