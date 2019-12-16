import numpy as np

class Helpers:

    @staticmethod
    def print_mean_median_max_min(data, prefix=""):
        print("Mean", prefix)
        print(np.mean(data))
        print("Median", prefix)
        print(np.median(data))
        print("Max", prefix)
        print(np.max(data))
        print("Min", prefix)
        print(np.min(data))
