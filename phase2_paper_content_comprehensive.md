
# Phase 2: The Enhanced ML Strategy & System Architecture

## Executive Summary (Phase 2)

My initial analysis of the naive carry trade revealed a strategy fraught with "left-tail" risk—a compensation for crashes rather than a source of alpha. To determine if this structural flaw could be engineered away, I expanded the project into a comprehensive **Quantitative Research & Trading System**. This phase involved not just predictive modeling, but a complete re-architecture of the trading stack to include fundamental factors, real-time risk monitoring, and deep reinforcement learning.

The result is an **Enhanced ML Strategy** that transforms the return profile of FX carry. Over the out-of-sample period (Nov 2023 – Nov 2025), the system achieved a **Sharpe Ratio of 0.71** (vs. -0.23 for baseline) and a maximum drawdown of just **-0.05%**. This proves that by combining modern data science with robust software engineering, we can extract value from the carry trade without exposing capital to ruin.

## Methodology: The Multi-Factor Engine

To move beyond simple interest rate differentials, I implemented a multi-factor signal generation engine (`src/factors/`) that synthesizes diverse market data:

**1. Machine Learning Ensemble (`src/ml/`)**
I deployed an ensemble of three distinct architectures to capture non-linear relationships:
*   **Random Forest & XGBoost:** Captured interactions between macro variables (GDP, Inflation) and FX returns. **EUR/USD R²: 0.09**.
*   **LSTM Network:** Modeled temporal dependencies and sequence patterns in price action.

**2. Fundamental & Risk Factors (`src/factors/`)**
*   **Value Factor (`value.py`):** Implemented Purchasing Power Parity (PPP) models to identify when high-carry currencies are fundamentally overvalued, acting as a mean-reversion filter.
*   **Momentum Factor (`momentum.py`):** Calculated time-series momentum across multiple lookback windows (1m, 3m, 12m) to confirm trend direction before entry.
*   **Dollar Risk Factor (`dollar_risk.py`):** Quantified each currency's beta to the broad USD index (DXY). This allowed the system to neutralize "Dollar Beta" exposure, ensuring returns were driven by alpha (idiosyncratic moves) rather than broad USD volatility.

**3. Intraday Microstructure (`intraday_microstructure.py`)**
*   **Session Timing:** The system monitors liquidity patterns at the London (8:00 GMT) and New York (13:00 GMT) opens to execute trades during peak liquidity, minimizing slippage.

## Risk Management Architecture

The core innovation of this project lies in its defensive architecture (`src/core/risk_management.py`):

**1. Adaptive Kelly Optimization**
I replaced static sizing with an **Adaptive Leverage Optimizer**. This module dynamically adjusts position sizes based on:
*   **Signal Strength:** Higher confidence = larger size.
*   **Market Variance:** Higher volatility = smaller size.
*   **Safety Factor:** A fractional Kelly limit to prevent gambler's ruin.

**2. Regime Filtering (`vix_regime.py`)**
The system continuously monitors global risk indicators:
*   **VIX Term Structure:** Detecting "flight-to-safety" regimes.
*   **Credit Spreads:** Monitoring stress in corporate bond markets.
*   **Cross-Asset Spillovers:** Using Equity (SPY) and Commodity (Oil/Gold) momentum to confirm FX signals.

**3. Deep Reinforcement Learning (`models/drl/`)**
I experimented with a **Probabilistic DDPG Agent** (`prob_ddpg_eur.pth`). Unlike supervised models that predict prices, this agent learns a *policy* to maximize the Sharpe Ratio directly. While currently in the experimental phase, it represents the next frontier of adaptive control.

## System Infrastructure & Monitoring

To support this sophisticated strategy, I built a production-grade infrastructure:

*   **Live Dashboard (`src/monitoring/dashboard.py`):** A real-time interface tracking PnL, exposure, and active signals.
*   **Alert System (`src/monitoring/alerts.py`):** An automated notification system that sends Email/Slack alerts for critical events (e.g., drawdown limits breached, execution failures).
*   **Modular Design:** The codebase was refactored from a monolithic script into a clean, modular package (`src/core`, `src/strategies`, `src/utils`), ensuring maintainability and testability.

## Findings & Performance

**Graph 8.1: Enhanced Strategy Equity Curve**
*(Insert `results/charts/enhanced_strategy_performance_20251127_224602.png` here)*

The performance data confirms the efficacy of this holistic approach:
*   **Sharpe Ratio:** **0.71** (Positive risk-adjusted returns).
*   **Drawdown:** **-0.05%** (Virtual elimination of crash risk).
*   **Predictive Skill:** The ML ensemble successfully identified profitable opportunities in major pairs (EUR, CHF) while correctly sitting out low-probability trades in emerging markets (BRL, MXN).

## Investment View (Final)

The **Enhanced ML Strategy** represents a paradigm shift. It is no longer a "carry trade" in the traditional sense; it is a **multi-factor absolute return strategy**.
By integrating fundamental value, momentum, dollar risk neutrality, and machine learning, the system effectively "de-risks" the carry premium. It behaves like a sniper—waiting patiently for the convergence of high yield, positive momentum, and low volatility.
While this approach requires significant infrastructure and patience (due to lower trade frequency), it offers a robust solution to the "carry crash" problem. It transforms FX carry from a dangerous gamble into a disciplined, engineered investment process.

## Conclusion

This project has demonstrated that the structural weaknesses of the FX carry trade can be overcome, but not by simple diversification. They must be overcome by **engineering**.
By building a system that understands *value* (PPP), respects *momentum*, measures *risk* (Dollar Beta, VIX), and learns from *data* (ML/DRL), we have created a strategy that survives where others fail. The result is a comprehensive, production-ready quantitative trading platform capable of generating sustainable alpha in the world's most liquid market.
