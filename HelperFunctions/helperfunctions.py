import pandas as pd
import numpy as np
import random
    

class HelperFunctions:
    
    def __init__(self, test_size):
        self.test_size = test_size
    
    def train_test_split(self, df):
    
        if isinstance(self.test_size, float):
            self.test_size = round(self.test_size * len(df))
    
        indices = df.index.tolist()
        test_indices = random.sample(population=indices, k=self.test_size)
        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)
        
        return train_df, test_df
    
    def calculate_accuracy(self, predictions, labels):
        predictions_correct = predictions == labels
        accuracy = predictions_correct.mean()
        
        return accuracy
    
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())