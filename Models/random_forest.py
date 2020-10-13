import random
import pandas as pd
import numpy as np
import os
import random

from decision_trees import decisiontree
from helperfunctions import HelperFunctions
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
    
                        ## RANDOM FOREST CLASS ##
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class randomforest:
    
    def __init__(self, n_bootstrap, n_trees, n_features, dt_max_depth, min_samples):
        self. n_bootstrap = n_bootstrap
        self.n_trees = n_trees
        self.n_features = n_features
        self.dt_max_depth = dt_max_depth
        self.min_samples = min_samples
    
    def Bootstrapping(self, df):
        bootstrap_indices = np.random.randint(low = 0, high = len(df), size = self.n_bootstrap)
        df_bootstrapped = df.iloc[bootstrap_indices]
        return df_bootstrapped
    
    def fit(self, df):
        forest = []
        for i in range(self.n_trees):
            df_bootstrapped = self.Bootstrapping(df)
            dt = decisiontree(counter = 0, min_samples = self.min_samples, max_depth = self.dt_max_depth, random_subspace = self.n_features)
            tree = dt.decision_tree_algorithm(df_bootstrapped)
            forest.append(tree)
        return forest
    
    def predict(self, test_df, forest, problemtype):
        df_predictions = {}
        for i in range(len(forest)):
            column_name = "tree_{}".format(i)
            dt = decisiontree(counter = 0, min_samples = 2, max_depth = self.dt_max_depth, random_subspace = self.n_features)
            predictions = dt.decision_tree_predictions(test_df, tree = forest[i])
            df_predictions[column_name] = predictions
        
        df_predictions = pd.DataFrame(df_predictions)
        if problemtype == 'c':
            rfpredictions = df_predictions.mode(axis=1)[0]
        else:
            rfpredictions = df_predictions.mean(axis=1)
        return rfpredictions
    
