import logging
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


class Yahoo():
    """
    Retrieves components, prices and sector from boursier.com and yfinance.
    """
    
    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:144.0) Gecko/20100101 Firefox/144.0'

    def get_universe(self, index_name="CAC40"):
        """
        Retrieves the constituents of the CAC40
        """

        try:
            df = self._scrape_boursier_cac40()
            df = self._retrieve_sectors_symbol(df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch universe: {e}")
            raise

    def get_prices(self, tickers, start_date):
        """
        Fetches historical prices from Yahoo Finance.
        """
        if not tickers:
            logger.warning("Empty ticker list provided.")
            return pd.DataFrame()
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                auto_adjust=True, 
                progress=False,
                threads=True
            )['Close']
            
            missing_cols = [t for t in tickers if t not in data.columns]
            if missing_cols:
                logger.warning(f"Missing data for: {missing_cols}")
            
            data.dropna(axis=1, how='all', inplace=True)
            
            return data
        except Exception as e:
            logger.critical(f"Critical failure in price fetch: {e}")
            raise

    def _scrape_boursier_cac40(self):
        """
        Scrapes CAC40 constituents from Boursier.com
        """
        url = "https://www.boursier.com/indices/composition/cac-40-FR0003500008,FR.html"
        headers = {'User-Agent': self.user_agent}
        
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        table = soup.find("table", class_="table--values")
        
        results = []
        for row in table.select("tr"):
            link = row.find("a", class_="name--wrapper")
            if link:
                name = link.get_text(strip=True)
                href = link['href']
                try:
                    ticker = href.split('-')[-1].split(',')[0]
                except IndexError:
                    continue
                
                results.append({"company": name, "ticker": ticker})
                
        return pd.DataFrame(results)

    def _retrieve_sectors_symbol(self, table):
        """
        Adds the sector and symbol from yfinance.
        """
        sectors = []
        symbols = []
        
        for t in table['ticker']:
            try:
                info = yf.Ticker(t).info
                sectors.append(info.get('sector', 'Unknown'))
                symbols.append(info.get('symbol', 'Unknown'))
            except Exception:
                sectors.append('Unknown')
                symbols.append('Unknown')
            
        table['sector'] = sectors
        table['symbol'] = symbols
        return table
