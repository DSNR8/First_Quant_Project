# Stochastic Portfolio Optimization & Stress Testing 📈

An end-to-end quantitative finance project focusing on dynamic asset allocation, risk management, and Monte Carlo simulations using Stochastic Differential Equations (SDEs).

## 🎯 Project Overview
This project constructs a diversified portfolio using European and US ETFs, optimizes it based on Modern Portfolio Theory (Markowitz), and rigorously tests its resilience through Monte Carlo simulations and Stress Testing models.

**Assets Analyzed:**
* `VUAA.DE` (Vanguard S&P 500 UCITS ETF - EUR)
* `AETF.AT` (Alpha ETF ATHEX ESG - EUR)
* `GLD` (SPDR Gold Shares - USD)
* `TLT` (iShares 20+ Year Treasury Bond ETF - USD)

## ⚙️ Key Features & Methodology

### 1. Constrained Portfolio Optimization
Calculates the optimal asset weights to maximize the **Sharpe Ratio**. To ensure realistic institutional allocation and avoid mathematical biases (e.g., over-allocating to historical bull-run assets), a strict $35\%$ maximum weight constraint is applied per asset.

### 2. Monte Carlo Simulation (Geometric Brownian Motion)
Projects 10,000 potential future portfolio paths over a 1-year horizon (252 trading days). The future value $S_t$ is modeled using the closed-form solution of the GBM Stochastic Differential Equation:
$$S_t = S_{t-1} \cdot e^{\left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} Z}$$
where $Z \sim \mathcal{N}(0,1)$.

### 3. Value at Risk (VaR) & Stress Testing
* **Historical VaR (95%):** Computes the expected maximum loss under normal market conditions.
* **Black Swan Stress Test:** Simulates a severe market crash by forcing the expected return ($\mu$) to $-20\%$ and doubling the volatility ($\sigma$). This reveals the "Stressed VaR", providing critical insight for risk management.

## 🚀 How to Run
Ensure you have the required libraries installed:
`pip install numpy pandas matplotlib scipy yfinance`

Run the main script:
`python main.py`