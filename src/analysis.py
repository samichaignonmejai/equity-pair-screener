import numpy as np
import pandas as pd
import statsmodels.api as sm
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
    MIN_OBSERVATIONS: int = 500          # Minimum overlapping days required
    P_VALUE_THRESHOLD: float = 0.05      # Significance level for Cointegration
    MIN_HALF_LIFE: float = 1.0           # Minimum days for half-life
    MAX_HALF_LIFE_DISPLAY: float = 1000.0 # Cap for display purposes
    JOHANSEN_DET_ORDER: int = 0          # Deterministic order for Johansen
    JOHANSEN_K_AR_DIFF: int = 1          # Lag difference


class PairsTradingAnalyzer:
    """
    Analyze stock pairs for mean-reversion properties.
    """
    
    def __init__(self, prices_df, config: StrategyConfig):
        self.prices = prices_df
        self.config = config

    def _calculate_half_life(self, spread):
        """
        Calculates the Half-Life of a mean-reverting spread.
        
        This assumes the spread follows an OU process:
        dx_t = theta (mu - x_t) dt + sigma dW_t
        
        We discretize this to:
        x_t - x_{t-1} = alpha + beta x_{t-1} + epsilon_t
        
        Where half-life is: HL = -frac{ln(2)}{beta}
        """
        if len(spread) < 100:
            return np.inf
            
        delta = np.diff(spread)
        lag = spread[:-1]
        
        X = sm.add_constant(lag)
        
        try:
            # Regress change (delta) against previous value (lag)
            model = sm.OLS(delta, X).fit()
            lambda_coef = model.params[1] # This is our beta (mean reversion speed)

            # If lambda >0, spread is diverging
            # if lambda = 0 ,spread is a Random Walk
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
        """
        Compute all the statistics for a given pair of tickers.
        """
    
        if ticker1 not in self.prices.columns or ticker2 not in self.prices.columns:
            logger.info(f"One of the tickers {ticker1} or {ticker2} not in price data.")
            return None

        # We compare prices on days where BOTH stocks traded.
        series_y = self.prices[ticker1].dropna()
        series_x = self.prices[ticker2].dropna()
        
        common_index = series_y.index.intersection(series_x.index)
        
        # Verify minimum observations
        if len(common_index) < self.config.MIN_OBSERVATIONS:
            logger.info(f"Not enough overlapping observations for {ticker1}-{ticker2}. Required: {self.config.MIN_OBSERVATIONS}, Found: {len(common_index)}")
            return None
            
        # Align data
        y = series_y.loc[common_index]
        x = series_x.loc[common_index]
        
        # Log-transformation: We use log prices so we are analyzing returns/percentages
        # rather than raw dollar amounts (for homoscedasticity).
        log_y, log_x = np.log(y), np.log(x)

        # EG Cointegration Test
        try:
            _, pvalue, _ = coint(log_y, log_x, trend='c', autolag='BIC')
            if pvalue > self.config.P_VALUE_THRESHOLD:
                return None
        except Exception as e:
            logger.debug(f"EG Test failed for {ticker1}-{ticker2}: {e}")
            return None

        # Calculate Hedge Ratio (Beta) via OLS
        # Equation: log(Y) = alpha + beta * log(X) + epsilon
        ols_model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
        beta = ols_model.params[1]
        
        # Construct the Spread
        spread_series = log_y - beta * log_x
        spread_values = spread_series.values

        # Johansen Test
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

        # Half-Life & Hurst
        half_life = self._calculate_half_life(spread_values)
        hurst = self._compute_hurst(spread_values)
        spread_std = np.std(spread_values)

        if half_life > self.config.MAX_HALF_LIFE_DISPLAY:
            hl_str = f">{int(self.config.MAX_HALF_LIFE_DISPLAY)}"
        else:
            hl_str = str(round(half_life, 1))

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
        """
        Returns the eligible pairs and the matching statistics.
        """
        results = []
        total_pairs = len(pairs_df)

        for idx, row in pairs_df.iterrows():
            res = self.analyze_pair(
                ticker1=row['Ticker1'],
                ticker2=row['Ticker2'],
                sector=row['Sector']
            )
            if res:
                results.append(res)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{total_pairs}. Found {len(results)} candidates.")

        results = pd.DataFrame(results)
        
        if not results.empty:
            results['sort_key'] = pd.to_numeric(
                results['half_life_days'].astype(str).str.replace('>', ''), 
                errors='coerce'
            )
            results = results.sort_values('sort_key').drop(columns=['sort_key'])
            
        return results