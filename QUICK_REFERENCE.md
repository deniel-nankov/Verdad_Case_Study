# FX Carry Strategy - Quick Reference Guide

## Key Formulas

### 1. Currency Excess Return
```
Excess Return(t) = Log Return(FX) + Interest Rate Differential

Where:
- Log Return(FX) = -ln(ExchRate(t) / ExchRate(t-1))
  (negative because higher rate = depreciation)
- Interest Rate Differential = (Foreign Rate - US Rate) / 252 / 100
  (daily rate from annualized rate)
```

### 2. Performance Metrics

**CAGR (Compound Annual Growth Rate)**
```python
cumulative_return = (1 + returns).prod()
n_years = len(returns) / 252
CAGR = (cumulative_return ^ (1 / n_years)) - 1
```

**Annualized Volatility**
```python
annual_vol = daily_std * sqrt(252)
```

**Sharpe Ratio**
```python
Sharpe = (mean_daily_return / std_daily_return) * sqrt(252)
```

**Maximum Drawdown**
```python
cumulative = (1 + returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
```

### 3. Information Coefficient
```python
IC = correlation(signal(t), forward_return(t+1))
```

## Strategy Construction

### Monthly Rebalancing Process

**Step 1:** Calculate interest rate differentials at month-end
```python
int_diff = foreign_rate - us_rate
```

**Step 2:** Rank currencies
```python
ranked = int_diff.sort_values(ascending=False)
```

**Step 3:** Assign positions
```python
Long:  Top 3 currencies (+1/3 weight each)
Short: Bottom 3 currencies (-1/3 weight each)
```

**Step 4:** Calculate strategy return
```python
strategy_return = sum(weight(i) * excess_return(i))
```

## Currency Codes & Characteristics

| Currency | Code | Typical Carry | Safe Haven | Characteristics |
|----------|------|---------------|------------|-----------------|
| Australian Dollar | AUD | High | No | Commodity-linked, risk-on |
| Brazilian Real | BRL | High | No | EM, high rates |
| Canadian Dollar | CAD | Medium | No | Commodity-linked |
| Swiss Franc | CHF | Low | Yes | Flight-to-quality |
| Euro | EUR | Low-Med | Sometimes | Large liquid market |
| British Pound | GBP | Medium | No | Developed market |
| Japanese Yen | JPY | Low | Yes | Funding currency |
| Mexican Peso | MXN | High | No | EM, commodity-linked |

## Exchange Rate Convention

**Important:** Higher exchange rate = MORE foreign currency per USD
- Increase in rate = Foreign currency DEPRECIATION
- Decrease in rate = Foreign currency APPRECIATION

From US investor perspective:
- We want foreign currency to APPRECIATE (rate goes DOWN)
- Hence: Return = -log(rate(t) / rate(t-1))

## Risk Characteristics by Carry Level

### High-Carry Currencies (BRL, MXN, AUD)
- Higher average returns
- Higher volatility
- Negative skewness (left tail)
- Positive equity correlation
- Worse in crises

### Low-Carry Currencies (JPY, CHF)
- Lower/negative average returns
- Lower volatility  
- Positive skewness (right tail)
- Negative equity correlation
- Better in crises (safe haven)

## Statistical Moments

**Skewness:**
- Negative: More left tail (large negative returns)
- Positive: More right tail (large positive returns)
- High-carry typically shows negative skew

**Kurtosis:**
- High kurtosis: Fat tails (extreme events)
- Indicates crash risk
- Carry strategies typically show excess kurtosis

## Regime Analysis Framework

### Volatility Regimes
```python
# Calculate rolling volatility
vol = equity_returns.rolling(21).std() * sqrt(252)

# Define regimes
high_vol = vol > median(vol)
low_vol = vol <= median(vol)

# Compare performance
perf_high_vol = strategy_returns[high_vol]
perf_low_vol = strategy_returns[low_vol]
```

### Expected Pattern
- **Low Vol:** Carry performs well (positive Sharpe)
- **High Vol:** Carry struggles (negative/low returns)

## Key Insights to Remember

### 1. Uncovered Interest Rate Parity (UIP)
**Theory:** High interest rate currencies should depreciate to offset carry gain
**Reality:** They don't depreciate enough → Carry premium exists
**Interpretation:** Market puzzle / risk premium

### 2. Carry as Risk Premium
Investors earn carry for bearing:
- Crash risk (negative skew)
- Liquidity risk
- Pro-cyclical exposure
- Currency crisis risk

### 3. Diversification Properties
- **Normal times:** Low equity correlation (good diversification)
- **Crisis times:** Positive equity correlation (poor diversification)
- **Net effect:** Fair-weather friend

### 4. Implementation Considerations
- Transaction costs (bid-ask spreads)
- Rollover costs/benefits
- Position sizing based on volatility
- Rebalancing frequency
- Currency selection (G10 vs EM)

## Common Pitfalls

1. **Look-ahead bias** - Using future information
2. **Survivorship bias** - Only currencies that survived
3. **Data snooping** - Overfitting to historical data
4. **Ignoring costs** - Transaction costs matter
5. **Regime blindness** - Not adjusting for market conditions

## Evaluation Checklist

### Data Quality
- [ ] Correct date alignment
- [ ] No look-ahead bias
- [ ] Proper handling of weekends/holidays
- [ ] Rates in correct units (annualized)

### Strategy Implementation
- [ ] Rebalancing on correct dates (month-end)
- [ ] Weights sum to zero (long-short)
- [ ] Returns properly calculated
- [ ] No position before first rebalance

### Analysis Completeness
- [ ] Summary statistics calculated
- [ ] Visualizations created
- [ ] Performance metrics computed
- [ ] Correlations analyzed
- [ ] Risk assessment done

### Memo Quality
- [ ] Clear executive summary
- [ ] Data and methodology described
- [ ] Key findings highlighted
- [ ] Charts included
- [ ] Investment view stated
- [ ] Risks discussed
- [ ] Proper formatting

## Further Reading

### Academic Papers
- Brunnermeier et al. "Carry Trades and Currency Crashes" (2008)
- Lustig et al. "Common Risk Factors in Currency Markets" (2011)
- Menkhoff et al. "Carry Trades and Global FX Volatility" (2012)

### Concepts to Explore
- Forward premium puzzle
- Dollar carry trade
- Currency momentum
- Value vs carry
- Volatility risk premium

---

**Remember:** This is an analysis tool. Your insights and interpretation matter most!
