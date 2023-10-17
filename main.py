from Architecture.architecture import PreprocessingMethod
from Architecture.ff import FeedForward

if __name__ == "__main__":
    ff = FeedForward(csv="AAPL.csv",window_size=100,prep_method=PreprocessingMethod.FF_ITERATIVE)
    ff.preprocess()
