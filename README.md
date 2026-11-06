# Portfolio Optimizer — Modern Portfolio Theory in Python

An implementation of Harry Markowitz's **Modern Portfolio Theory (MPT)** using 5 years of historical US equity data (`2021–2025`), Monte Carlo simulation (`10,000` random allocations), and exact constrained optimization (`scipy.optimize` SLSQP).

## What It Does
* Downloads 5 years of daily closing prices for 6 US stocks (`AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `JNJ`) via `yfinance`
* Computes annualized returns ($\times 252$), annualized volatility ($\times \sqrt{252}$), and the asset correlation matrix
* Simulates 10,000 random portfolios (Monte Carlo) to map out the **Efficient Frontier**
* Solves for the exact **Maximum Sharpe Ratio** portfolio using `scipy.optimize.minimize` (SLSQP)
* Compares cumulative wealth growth of `$1` invested in the optimized portfolio vs. an equal-weight (`1/6`) benchmark

## Results (6 US Stocks, 2021–2025)

| Portfolio | $1 Grows To | Max Sharpe |
| :--- | :--- | :--- |
| **Optimized (`AMZN` 55%, `JNJ` 31%, `NVDA` 14%)** | **$4.07** | **1.49** |
| **Equal Weight (1/6 each)** | **$3.30** | **—** |

* **Diversification in action:** Even though `JNJ` had the lowest annual return (`9.9%`), the optimizer allocated **31%** to it because its near-zero correlation with tech stocks (`0.01` with `AMZN`, `-0.10` with `NVDA`) dampens overall portfolio variance ($\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$).
* **Honest caveat:** Optimization on historical means and covariances is **in-sample** and amplifies estimation error. Future iterations can incorporate out-of-sample walk-forward testing, weight caps, or shrinkage estimators.

## Efficient Frontier Chart
![Efficient Frontier](images/efficient_frontier.png)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook 01_download_data.ipynb
