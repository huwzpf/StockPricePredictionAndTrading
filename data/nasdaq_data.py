import pandas as pd
import yfinance as yf
import random
from datetime import date
import os

class NasdaqData:
    
    def __init__(self,start : date,end: date):
        self.symbols = []
        self.all_symbols = self._load_all_symbols()
        self.start = start
        self.end = end
        
    def _load_all_symbols(self) -> dict:
        df = pd.read_csv("data/symbols.csv")
        return df.set_index('Symbol')['Company Name'].to_dict()
    
    def download_nasdaq_data(self,n=0):
        """
        Download csv for `self.symbols` companies for `self.start` to `self.end` period.
        
        @param n: If overridenc choose `n` random companies symbols to download instead.
        """
        if not os.path.exists(f"data/companies/{self.start}:{self.end}"):
            os.makedirs(f"data/companies/{self.start}:{self.end}")
        
        if n > 0:
            random_symbols = random.sample(list(self.all_symbols.keys()), n)
            self.symbols = list(random_symbols)

        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start, end=self.end)
            closing_prices = data['Close']
            closing_prices.to_csv(f"data/companies/{self.start}:{self.end}/{symbol}.csv")
    
    def add_symbols(self,symbols : list):
        """
        Manualy choose which companies will be downloaded.
        
        @param symbols: list of comapnies symbols to add to current list
        """
        self.symbols = list(set(self.symbols.extend(symbols)))
        
    def merge_csv(self,n=0):
        """
        Merge all csvs from `data/comapnies` folder and normalize relative to each column.
        
        @param n: If overriden choose `n` csvs and merge them together instead.
        """
        path = f"data/companies/{self.start}:{self.end}"
        merge_df = pd.DataFrame()
        if os.path.exists(path) and os.path.isdir(path):
            csvs = os.listdir(path)
            if n>0:
                csvs = [csv for csv in csvs if "merged" not in csv]
                csvs = random.sample(csvs, n)
            else:
                csvs = [csv for csv in csvs if "merged" not in csv]
            
        for csv in csvs:
                symbol= csv[:-4]
                df = pd.read_csv(os.path.join(path,csv))
                # Normalize
                min_val = df['Close'].min()
                max_val = df['Close'].max()
                df['Close'] = (df['Close'] - min_val) / (max_val - min_val)
                merge_df[symbol] = df['Close']

        merge_df.to_csv(os.path.join(path,f"merged_{len(csvs)}.csv"),index=False)
        