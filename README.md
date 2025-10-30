# Verdad FX Carry Strategy - Technical Case Study

## Overview
This repository contains a comprehensive analysis of FX carry strategies using daily data from 2000-2025 for 8 major currencies.

## Files

### 1. `fx_carry_analysis.ipynb` - Main Analysis Notebook
Complete Jupyter notebook with all analysis including:
- Data loading and preprocessing
- Currency excess returns calculation
- Risk-return analysis
- Monthly rebalanced long-short strategy construction
- Performance metrics (CAGR, Sharpe, Max Drawdown)
- Currency-equity correlation analysis
- Framework for evaluating predictive signals

### 2. `verdad_fx_case_study_data.csv` - Data File
Daily data containing:
- Exchange rates for 8 currencies (AUD, BRL, CAD, CHF, EUR, GBP, JPY, MXN)
- Spot interest rates
- S&P 500 Total Return index
- US Federal Funds Rate

### 3. `memo_template.md` - Investment Memo Template
Pre-structured memo template for the written deliverable. Fill in the bracketed sections with results from the notebook.

## Getting Started

### Step 1: Run the Notebook
```bash
# Open the notebook
jupyter notebook fx_carry_analysis.ipynb

# Or if using VS Code
# Just open the .ipynb file
```

### Step 2: Execute All Cells
Run all cells in order to:
1. Load and parse the data
2. Calculate excess returns
3. Build the carry strategy
4. Generate all visualizations
5. Compute performance metrics

### Step 3: Complete the Memo
Use `memo_template.md` as a guide:
1. Copy key statistics from notebook output
2. Include relevant charts
3. Add your analysis and interpretation
4. Export as PDF for submission

## Key Analysis Components

### 1. Currency Excess Returns
Formula: `Excess Return = ΔFX + (Foreign Rate - US Rate)`

Each currency's excess return captures both appreciation/depreciation and the carry premium.

### 2. Carry Strategy
- **Universe:** 8 currencies
- **Ranking:** By interest rate differential (Foreign - US)
- **Positions:** Long top 3, Short bottom 3
- **Weights:** Equal weight (1/3 per position)
- **Rebalancing:** Monthly

### 3. Performance Metrics
- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Correlation to Equities

### 4. Risk Analysis
- Distribution analysis (skewness, kurtosis)
- Tail risk assessment
- Drawdown analysis
- Regime-dependent performance

## Key Questions Answered

1. **Is carry risk compensation or free lunch?**
   - Analysis of skewness, kurtosis, and tail risk
   - Comparison of risk-adjusted returns
   - Regime-dependent performance

2. **Strategy Performance**
   - Full period metrics (2000-2025)
   - Crisis period analysis
   - Comparison to equities

3. **Currency-Equity Relationships**
   - Individual currency correlations
   - High-carry vs low-carry patterns
   - Diversification properties

4. **Investment Appeal**
   - Benefits and risks
   - Optimal portfolio allocation
   - Implementation considerations

5. **Predictive Signals**
   - Framework for evaluation
   - Example with equity volatility
   - Statistical testing methodology

## Expected Output

### Deliverables
1. **Jupyter Notebook** with all code and visualizations ✓
2. **Investment Memo** (1-2 pages) - Use template provided
3. **Key Charts** - Already generated in notebook

### Time Estimate
- Notebook execution: 10-15 minutes
- Memo writing: 2-3 hours
- Total: 4-6 hours

## Technical Requirements

### Python Packages
```python
pandas
numpy
matplotlib
seaborn
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn
```

## Tips for Success

1. **Run cells sequentially** - Each cell builds on previous ones
2. **Review all visualizations** - They tell the story
3. **Check summary statistics** - Numbers support conclusions
4. **Compare regimes** - High vol vs low vol performance
5. **Think critically** - What does the data really say?

## Key Insights to Highlight

### In Your Memo

1. **Skewness & Kurtosis**
   - High-carry currencies show negative skew
   - Evidence of crash risk
   - Not a free lunch!

2. **Correlation Patterns**
   - Higher-carry = higher equity correlation
   - Pro-cyclical risk
   - Poor crisis diversification

3. **Performance Attribution**
   - Positive long-term returns
   - Significant drawdowns in crises
   - Sharpe ratio vs equities

4. **Investment Implications**
   - Size appropriately (5-10% allocation)
   - Combine with defensive strategies
   - Active risk management essential

## Common Questions

**Q: Why does the strategy lose money in some periods?**
A: Carry strategies are short volatility - they suffer when risk-off periods cause funding currencies to appreciate.

**Q: Which currencies are typically high-carry?**
A: Historically BRL, MXN, AUD (higher rates, commodity-linked)

**Q: Which are low-carry/safe havens?**
A: JPY, CHF, sometimes EUR (lower rates, flight-to-quality)

**Q: How often does the strategy rebalance?**
A: Monthly, at month-end

**Q: What about transaction costs?**
A: Not explicitly modeled, but mentioned as implementation consideration

## Contact & Support

For questions about the analysis or methodology, refer to:
- Notebook comments and docstrings
- Memo template structure
- Academic literature on carry trade

## License & Attribution

This is a technical case study for interview purposes.

---

**Good luck with your analysis!** 🚀
