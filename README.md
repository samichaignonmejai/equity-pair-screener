# French Equity Pair Trading Screener

## Abstract

This repository implements a quantitative framework for identifying mean-reverting equity pairs within the CAC 40 index. The methodology reproduces and extends the screening procedure described by Figuerola‐Ferretti et al. (2018) in "Pairs‐trading and spread persistence in the European stock market".

The screener utilizes a multi-stage filtering process combining cointegration tests (Engle-Granger, Johansen) with mean-reversion metrics (Ornstein-Uhlenbeck Half-Life, Hurst Exponent) to robustly identify statistical arbitrage opportunities.

## Key Features


- Constraining the pairing with similar GICS sectors
- Engle-Granger and Johansen test to verify cointegration
- Mean reversion spread persistence checked with the OU process half-life
- Hurst exponent to differentiate from a random walk

## Project Structure
```
├── src/
│   ├── analysis.py
│   ├── data.py
│   └── processing.py
├── tutorial.ipynb
└── README.md
```

## Usage

The core workflow is demonstrated in tutorial.ipynb


## Roadmap

[ ] Backtesting Engine: Implement event-driven backtesting with transaction costs.

[ ] Visualization: Add spread plotting and Z-score entry/exit signal visualization.

[ ] Universe Selection: Abstract the data provider to support indices beyond the CAC 40 (e.g., DAX, EuroStoxx).

## References

Figuerola‐Ferretti, I., Paraskevopoulos, I., & Tang, T. (2018). Pairs‐trading and spread persistence in the European stock market. Journal of Futures Markets, 38(9), 998-1023. DOI: 10.2139/ssrn.3067442