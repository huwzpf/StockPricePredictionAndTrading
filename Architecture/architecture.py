from DataAPI.data_prep import Data
from enum import Enum
import pandas as pd

class PreprocessingMethod(Enum):
    FF_ITERATIVE = 0,
    FF_DIRECT = 1,
    FF_VECTOR = 2
    
class DataSubset(Enum):
    TRAIN = 0,
    VALIDATE = 1,
    TEST = 2

class Architecture:
    def __init__(self,csv : str,window_size : int,training_size=0.7, validation_size=0.2, test_size = 0.1):
        self.data = Data(csv)
        self.window_size = window_size
        self.X = [[],[],[]]
        self.Y = [[],[],[]]
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        
    def preprocess(self):
        """
        This function handles data preprocessing and should be implemented for every architecture.
        """
        print("Preprocessing not implemented!")
        
    def split(self):
        """
        
        """
        n_train = self.window_size * self.training_size
        n_validate = self.window_size * self.validation_size
        n_test = self.window_size * self.test_size
        
        df = pd.read_csv(self.data.out_path)
        
        for index in range(len(df)):
            row = list(df.iloc[index])
            self.X[DataSubset.TRAIN] = row[:n_train]
            self.Y[DataSubset.TRAIN] = row [n_train]
            self.X[DataSubset.VALIDATE] = row[n_train + 1: n_train + n_validate]
            self.Y[DataSubset.VALIDATE] = row[n_train + n_validate]
            self.X[DataSubset.TEST] = row[-n_test:-1]
            self.Y[DataSubset.TEST] = row[-1]
        
        
    def create_model(self):
        """
        
        """
        pass
    
    def fit(self):
        """
        
        """
        pass
       