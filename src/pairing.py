import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from typing import Tuple


logger = logging.getLogger(__name__)


def half_life_spread(spread):
    """
    Calculates the Half-Life of a mean-reverting spread using the Ornstein-Uhlenbeck process.

    Formula:
        dx_t = theta * (mu - x_t) * dt + sigma * dW_t
        Discretized: x_t - x_{t-1} = alpha + beta * x_{t-1} + epsilon
        Half-Life = -ln(2) / beta

    Args:
        spread (np.ndarray): Time series array of the spread.

    Returns:
        float: Half-life in days (or time units). Returns np.inf if not mean-reverting.
    """
    if len(spread) < 100:
        logger.debug("Spread length insufficient for reliable half-life calculation (<100).")
        return np.inf

    # Prepare regresson
    spread_diff = np.diff(spread)
    spread_lag = spread[:-1]
    X = sm.add_constant(spread_lag)

    try:
        # Fit OLS: Y = spread_diff, X = [1, spread_lag]
        model = sm.OLS(spread_diff, X)
        res = model.fit()
        beta = res.params[1]

        # Mean Reversion: Beta must be negative.
        # If Beta >= 0, the process is explosive or a random walk (non-stationary).
        if beta >= -1e-8:
            return np.inf

        half_life = -np.log(2) / beta
        
        # Cap min value to 1.0 to avoid numerical noise issues
        return max(half_life, 1.0)

    except Exception as e:
        logger.warning(f"Half-life calculation failed: {e}")
        return np.inf


class Pairing:
    """
    Methods:
        generate_pairs_by_sector: Generates all unique stock pairs within the same sector.
    """

    @staticmethod
    def generate_pairs_by_sector(constituents):
        """
        Generates all unique combinations of stocks within the same sector.

        Args:
            constituents (pd.DataFrame): Must contain columns ['sector', 'symbol', 'company'].

        Returns:
            pd.DataFrame: DataFrame containing all unique stock pairs within the same sector.
        """
        required_cols = {'sector', 'symbol', 'company'}
        if not required_cols.issubset(constituents.columns):
            missing = required_cols - set(constituents.columns)
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        all_pairs = []

        for sector, group in constituents.groupby('sector'):
            n_stocks = len(group)
            
            if n_stocks < 2:
                continue

            # Generate unique pairs
            pairs = list(combinations(group.itertuples(index=False), 2))
            
            for p1, p2 in pairs:
                all_pairs.append({
                    'Sector': sector,
                    'Stock1': p1.company,
                    'Ticker1': p1.symbol,
                    'Stock2': p2.company,
                    'Ticker2': p2.symbol
                })

        pairs = pd.DataFrame(all_pairs)

        if pairs.empty:
            logger.warning("No pairs could be generated. Check sector data.")
            return pd.DataFrame(), pd.DataFrame()

        return pairs