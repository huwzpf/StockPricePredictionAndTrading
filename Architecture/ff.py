from Architecture.architecture import PreprocessingMethod, Architecture
import pandas as pd
import numpy as np

# TODO:
# Implement horizon

class FeedForward(Architecture):
    """
    https://www.youtube.com/watch?v=OV39MtxOlFk
    """
    def __init__(self,csv : str,window_size : int, prep_method : PreprocessingMethod):
        super().__init__(csv)
        self.window_size = window_size
        if prep_method == PreprocessingMethod.FF_ITERATIVE:
            self.prep_method = self._preprocess_iterative
        elif prep_method == PreprocessingMethod.FF_DIRECT:
            self.prep_method = self._preprocess_direct
        else:
            self.prep_method = self._preprocess_vector
        
    def preprocess(self):
        self.prep_method()
    
    def _preprocess_iterative(self):
        """
        
        """
        df = pd.read_csv(self.data.in_path)
        data = df['close'].to_numpy()
        windowed_data = np.lib.stride_tricks.sliding_window_view(data[::-1], window_shape=(self.window_size,))
        df_windowed = pd.DataFrame(windowed_data)
        df_windowed.to_csv(self.input.out_path)
        
    def _preprocess_direct(self):
        """
        
        """
        print("Not implemented yet.")
        
    def _preprocess_vector(self):
        """
        
        """
        print("Not implemented yet.")
        
    