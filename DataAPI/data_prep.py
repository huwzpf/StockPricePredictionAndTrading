import numpy as np
import pandas as pd
from DataAPI import nasdaqdatalink

class Data:
    """
        This class handles raw time series data.
    """
    def __init__(self,csv=""):
        self.csv = csv
        self.in_path = "input/" + csv
        self.out_path = "output/" + csv
 
    def download(self,company : str,start : str,end,dataset='WIKI/PRICES'):
        """
        This function downloads data from NASDAQ API and save it in `.csv` format.
        
        @param: company : company code
        @start: start date in YYYY-MM-DD
        @end: end date in YYYY-MM-DD
        @dataset: dataset name         
        """
        data = nasdaqdatalink.get_table(dataset, qopts = { 'columns': ['date', 'close'] }, ticker = [company], date = { 'gte': start, 'lte': end })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        self.csv = f"{company}.csv"
        self.in_path = "input/" + self.csv
        self.out_path = "output" + self.csv
        df.to_csv(self.in_path)
        
    def normalize(self):
        pass
    
    def clean(self):
        pass