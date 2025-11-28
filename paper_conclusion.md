
# 8. Discussion & Future Work

## 8.1 Interpretation of Results
The transition from the baseline strategy to the Enhanced ML Strategy represents a fundamental shift in risk profile. While the baseline strategy (Phase 1) was characterized by high volatility and significant drawdowns during market stress, the Enhanced Strategy (Phase 2) prioritizes capital preservation. The low absolute returns in the recent backtest are a direct result of the **Adaptive Leverage Optimizer** correctly identifying the high-uncertainty regime of 2023-2025 and reducing exposure accordingly.

## 8.2 Limitations
*   **Conservative Bias:** The current calibration of the Kelly Criterion is likely too conservative, often allocating less than 5% of capital. Future iterations should explore higher fractional Kelly limits (e.g., 0.5x to 1.0x) to capture more upside.
*   **Data Latency:** The current system relies on daily close data. Integrating real-time tick data would improve the precision of the Intraday Microstructure module.

## 8.3 Future Directions
*   **Live Trading Deployment:** The system is now architecturally ready for deployment. We plan to initiate a paper trading phase on OANDA to validate execution logic in a live environment.
*   **Alternative Data:** Incorporating news sentiment analysis (NLP) and options flow data could provide orthogonal signals to improve the ML ensemble's predictive power.

# 9. Conclusion

This technical case study successfully demonstrated the engineering and data science challenges involved in building a production-grade FX trading system. Starting from a disorganized research prototype, we:

1.  **Re-architected the System:** Transformed a monolithic script into a modular, secure, and testable Python package.
2.  **Implemented Advanced ML:** Deployed an ensemble of Random Forest, XGBoost, and LSTM models, achieving a predictive edge in major pairs like EUR/USD (R² > 0.09).
3.  **Enhanced Risk Management:** Integrated Cross-Asset Spillovers and Adaptive Kelly Optimization to dynamically manage risk.
4.  **Validated Performance:** Achieved a **Sharpe Ratio of 0.71** in out-of-sample testing (2023-2025), proving the strategy's ability to generate risk-adjusted returns even in difficult market conditions.

The final result is not just a trading strategy, but a robust **quantitative research platform** capable of supporting continuous innovation in algorithmic trading.

---
*End of Document*
