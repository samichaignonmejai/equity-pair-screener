import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
import logging
from hurst import compute_Hc 
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    MIN_OBSERVATIONS: int = 500
    P_VALUE_THRESHOLD: float = 0.05
    MIN_HALF_LIFE: float = 1.0
    MAX_HALF_LIFE_DISPLAY: float = 1000.0
    JOHANSEN_DET_ORDER: int = 0
    JOHANSEN_K_AR_DIFF: int = 1
    ENTRY_Z_SCORE: float = 3.0
    EXIT_Z_SCORE: float = 0.0
    TRANSACTION_COST_BPS: float = 5.0
    RISK_FREE_RATE: float = 0.02
    ROLLING_WINDOW_DAYS: int = 252


class PairsTradingAnalyzer:
    """
    Analyze stock pairs for mean-reversion properties and backtest strategies.
    """
    
    def __init__(self, prices_df, config: StrategyConfig):
        self.prices = prices_df
        self.config = config

    def _calculate_half_life(self, spread):
        if len(spread) < 100:
            return np.inf
        delta = np.diff(spread)
        lag = spread[:-1]
        X = sm.add_constant(lag)
        try:
            model = sm.OLS(delta, X).fit()
            lambda_coef = model.params[1]
            if lambda_coef >= -0.0000001:
                return np.inf
            half_life = -np.log(2) / lambda_coef
            return max(half_life, self.config.MIN_HALF_LIFE)
        except Exception:
            return np.inf

    def _compute_hurst(self, spread):
        try:
            H, _, _ = compute_Hc(spread, kind='price', simplified=True)
            return round(H, 3)
        except Exception:
            return np.nan

    def analyze_pair(self, ticker1, ticker2, sector):
        if ticker1 not in self.prices.columns or ticker2 not in self.prices.columns:
            logger.info(f"One of the tickers {ticker1} or {ticker2} not in price data.")
            return None

        series_y = self.prices[ticker1].dropna()
        series_x = self.prices[ticker2].dropna()
        common_index = series_y.index.intersection(series_x.index)
        
        if len(common_index) < self.config.MIN_OBSERVATIONS:
            return None
            
        y = series_y.loc[common_index]
        x = series_x.loc[common_index]
        log_y, log_x = np.log(y), np.log(x)

        try:
            _, pvalue, _ = coint(log_y, log_x, trend='c', autolag='BIC')
            if pvalue > self.config.P_VALUE_THRESHOLD:
                return None
        except Exception:
            return None

        ols_model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
        beta = ols_model.params[1]
        
        spread_values = (log_y - beta * log_x).values
        
        # Johansen
        johansen_rank = 0
        try:
            det_order = -1 if self.config.JOHANSEN_DET_ORDER == 0 else 0
            vecm_res = select_coint_rank(
                pd.DataFrame({'y': log_y, 'x': log_x}), 
                det_order=det_order, 
                k_ar_diff=self.config.JOHANSEN_K_AR_DIFF,
                signif=0.05
            )
            johansen_rank = vecm_res.rank
        except Exception:
            pass

        half_life = self._calculate_half_life(spread_values)
        hurst = self._compute_hurst(spread_values)
        spread_std = np.std(spread_values)

        hl_str = f">{int(self.config.MAX_HALF_LIFE_DISPLAY)}" if half_life > self.config.MAX_HALF_LIFE_DISPLAY else str(round(half_life, 1))

        return {
            'pair': f"{ticker1}-{ticker2}",
            'company1': ticker1, 
            'company2': ticker2,
            'sector': sector,
            'p_value_EG': round(pvalue, 5),
            'beta': round(beta, 4),
            'half_life_days': hl_str,
            'hurst': hurst,
            'spread_std': round(spread_std, 5),
            'johansen_rank': johansen_rank,
            'n_obs': len(spread_values)
        }

    def run_batch_analysis(self, pairs_df):
        results = []
        total_pairs = len(pairs_df)

        for idx, row in pairs_df.iterrows():
            res = self.analyze_pair(row['Ticker1'], row['Ticker2'], row['Sector'])
            if res:
                results.append(res)
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{total_pairs}. Found {len(results)} candidates.")

        results = pd.DataFrame(results)
        if not results.empty:
            results['sort_key'] = pd.to_numeric(
                results['half_life_days'].astype(str).str.replace('>', ''), errors='coerce'
            )
            results = results.sort_values('sort_key').drop(columns=['sort_key'])
            
        return results

    def _calculate_metrics(self, backtest_df):
        """
        computes annualized performance and risk metrics.
        """
        returns = backtest_df['Strategy_Returns']
        equity_curve = backtest_df['Cumulative_PnL']
        
        total_return = equity_curve.iloc[-1] - 1
        days = (backtest_df.index[-1] - backtest_df.index[0]).days
        years = days / 365.25
        
        if years > 0:
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = 0.0

        daily_std = returns.std()
        ann_volatility = daily_std * np.sqrt(252)
        
        rf_daily = (1 + self.config.RISK_FREE_RATE) ** (1/252) - 1
        excess_returns = returns - rf_daily
        
        if daily_std > 0:
            sharpe_ratio = (excess_returns.mean() / daily_std) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        positions = backtest_df['Position']
        trade_returns = []
        current_ret = 0
        in_trade = False
        
        for i in range(1, len(positions)):
            prev = positions.iloc[i-1]
            curr = positions.iloc[i]
            ret = returns.iloc[i]
            
            if prev == 0 and curr != 0:
                in_trade = True
                current_ret = (1 + ret)
            elif prev != 0:
                current_ret *= (1 + ret)
                if curr == 0 or curr != prev:
                    trade_returns.append(current_ret - 1)
                    in_trade = False
                    if curr != 0:
                        in_trade = True
                        current_ret = 1.0

        trade_returns = np.array(trade_returns)
        n_trades = len(trade_returns)
        win_rate = np.sum(trade_returns > 0) / n_trades if n_trades > 0 else 0
        
        gross_profit = np.sum(trade_returns[trade_returns > 0])
        gross_loss = np.abs(np.sum(trade_returns[trade_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return {
            "Annualized Return": f"{cagr:.2%}",
            "Annualized Volatility": f"{ann_volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Total Trades": n_trades
        }

    def backtest_pair(self, ticker1, ticker2):
        """
        Runs the backtest, prints metrics, and plots the results.
        """
        # 1. Data Prep
        if ticker1 not in self.prices.columns or ticker2 not in self.prices.columns:
            logger.error("Tickers not found.")
            return
            
        series_y = self.prices[ticker1].dropna()
        series_x = self.prices[ticker2].dropna()
        common = series_y.index.intersection(series_x.index)
        
        if len(common) < self.config.MIN_OBSERVATIONS:
            logger.error("Not enough data.")
            return

        y = series_y.loc[common]
        x = series_x.loc[common]
        log_y, log_x = np.log(y), np.log(x)

        # rolling window mean and beta
        window = self.config.ROLLING_WINDOW_DAYS

        rolling_cov = log_y.rolling(window).cov(log_x)
        rolling_var = log_x.rolling(window).var()

        rolling_beta = rolling_cov / rolling_var

        rolling_mean_y = log_y.rolling(window).mean()
        rolling_mean_x = log_x.rolling(window).mean()
        rolling_spread = rolling_mean_y - rolling_beta * rolling_mean_x
        
        spread = log_y - (rolling_spread + rolling_beta * log_x)
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std()
        z_score = (spread - spread_mean) / spread_std

        valid_idx = z_score.dropna().index
        z_score = z_score.loc[valid_idx]
        rolling_beta = rolling_beta.loc[valid_idx]

        positions = pd.Series(0, index=z_score.index)
        entry, exit_ = self.config.ENTRY_Z_SCORE, self.config.EXIT_Z_SCORE
        
        curr = 0
        pos_vals = []
        for z in z_score.values:
            if curr == 0:
                if z > entry: curr = -1
                elif z < -entry: curr = 1
            elif curr == 1 and z >= exit_: curr = 0
            elif curr == -1 and z <= exit_: curr = 0
            pos_vals.append(curr)
        positions[:] = pos_vals

        # computes the PnL

        ret_x = y.pct_change().loc[valid_idx].fillna(0)
        ret_y = y.pct_change().loc[valid_idx].fillna(0)

        lagged_beta = rolling_beta.shift(1).fillna(method='bfill')
        spread_ret = ret_y - lagged_beta * ret_x
        strat_ret = positions.shift(1).fillna(0) * spread_ret
        
        trades = positions.diff().abs().fillna(0)
        costs = trades * (self.config.TRANSACTION_COST_BPS / 10000)
        
        net_ret = strat_ret - costs
        cum_pnl = (1 + net_ret).cumprod()

        df = pd.DataFrame({
            'Z_Score': z_score,
            'Position': positions,
            'Strategy_Returns': net_ret,
            'Cumulative_PnL': cum_pnl
        })

        metrics = self._calculate_metrics(df)
        
        print("-" * 30)
        print(f"Backtest Results: {ticker1} vs {ticker2}")
        print("-" * 30)
        for k, v in metrics.items():
            print(f"{k:<20}: {v}")
        print("-" * 30)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        
        # pnl of the strategy
        axes[0].plot(cum_pnl.index, cum_pnl, label='Strategy PnL', color='blue', linewidth=0.7)
        axes[0].set_title(f"Cumulative Return")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # spread and signals
        axes[1].plot(z_score.index, z_score, label='Spread', color='blue', linewidth=0.7)
        axes[1].axhline(entry, color='red', linestyle='--', label='Short Threshold')
        axes[1].axhline(-entry, color='green', linestyle='--', label='Long Threshold')
        axes[1].axhline(exit_, color='black', linewidth=1)
        axes[1].fill_between(z_score.index, entry, z_score, where=(z_score > entry), color='red', alpha=0.1)
        axes[1].fill_between(z_score.index, -entry, z_score, where=(z_score < -entry), color='green', alpha=0.1)
        axes[1].set_title("Historical Spread (Z-Score)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()