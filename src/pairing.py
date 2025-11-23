import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from typing import Tuple, List, Optional

# Configure module-level logger
logger = logging.getLogger(__name__)

# ==========================================
# 1. Math & Statistics Utilities
# ==========================================

def half_life_spread(spread: np.ndarray) -> float:
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
    # Data Integrity Check
    if len(spread) < 100:
        logger.debug("Spread length insufficient for reliable half-life calculation (<100).")
        return np.inf

    # Prepare regressors
    spread_diff = np.diff(spread)
    spread_lag = spread[:-1]
    
    # Add constant (intercept) to the model
    X = sm.add_constant(spread_lag)

    try:
        # Fit OLS: Y = spread_diff, X = [1, spread_lag]
        model = sm.OLS(spread_diff, X)
        res = model.fit()
        beta = res.params[1]  # The coefficient of x_{t-1}

        # Mean Reversion Logic: Beta must be negative.
        # If Beta >= 0, the process is explosive or a random walk (non-stationary).
        if beta >= -1e-8:
            return np.inf

        half_life = -np.log(2) / beta
        
        # Cap min value to 1.0 to avoid numerical noise issues
        return max(half_life, 1.0)

    except Exception as e:
        logger.warning(f"Half-life calculation failed: {e}")
        return np.inf


# ==========================================
# 2. Pair Generation Logic
# ==========================================

class Pairing:
    """
    Service class responsible for generating candidate pairs from a stock universe.
    Encapsulates logic for different pairing strategies (Sector, Cluster, PCA).
    """

    @staticmethod
    def generate_pairs_by_sector(constituents: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates all unique combinations of stocks within the same sector.

        Args:
            constituents (pd.DataFrame): Must contain columns ['sector', 'symbol', 'company'].

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (Summary Table, Pairs Table)
        """
        # 1. Input Validation (Defensive Programming)
        required_cols = {'sector', 'symbol', 'company'}
        if not required_cols.issubset(constituents.columns):
            missing = required_cols - set(constituents.columns)
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        # logger.info(f"Generating pairs from {len(constituents)} constituents...")

        all_pairs = []
        summary_stats = []

        # 2. Group & Process
        # Using groupby is more memory efficient than iterating rows manually
        for sector, group in constituents.groupby('sector'):
            n_stocks = len(group)
            
            if n_stocks < 2:
                continue

            # Generate unique pairs (nCr)
            # using .itertuples(index=False) is much faster than .iterrows()
            pairs = list(combinations(group.itertuples(index=False), 2))
            
            # 3. Build Metadata
            for p1, p2 in pairs:
                all_pairs.append({
                    'Sector': sector,
                    'Stock1': p1.company,
                    'Ticker1': p1.symbol,
                    'Stock2': p2.company,
                    'Ticker2': p2.symbol
                })
            
            summary_stats.append({
                'Sector': sector,
                '# Stocks': n_stocks,
                '# Pairs': len(pairs)
            })

        # 4. Final Output Construction
        pairs_df = pd.DataFrame(all_pairs)
        summary_df = pd.DataFrame(summary_stats)

        # Handle empty results gracefully
        if pairs_df.empty:
            logger.warning("No pairs could be generated. Check sector data.")
            return pd.DataFrame(), pd.DataFrame()

        # logger.info(f"Generated {len(pairs_df)} pairs across {len(summary_df)} sectors.")
        return pairs_df