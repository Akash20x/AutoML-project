import os
import numpy as np
import pandas as pd
import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle
import os
os.chdir("C:\\Users\\jacob\\OneDrive\\Documents\\Infovio Data Science Internship\\autoML full")
import sys
sys.path.append("C:\\Users\\jacob\\OneDrive\\Documents\\Infovio Data Science Internship\\autoML full")
from preprocessingoop import Preprocessing
os.chdir("C:\\Users\\jacob\\OneDrive\\Documents\\Infovio Data Science Internship\\autoML\\Test datasets")

df = pd.read_csv('breastcancerwisc.csv')

#concatenate a column of ones for the intercept

class SVM:
         
    def __init__(self, reg_strength, learning_rate):
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
                     
    def Cost(self, W, X, Y):
        n = X.shape[0]
        distances = 1 - Y*(np.dot(X, W))
        distances[distances < 0] = 0 #distances which are less than 0 are set to zero (max(0, distance))
        hinge_loss = self.reg_strength * (np.sum(distances)/n)
        cost = 1/2 * np.dot(W, W) + hinge_loss
        return cost
        
    def GradientCost(self, W, X_batch, Y_batch):
        #if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
            
        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))
        
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.reg_strength * Y_batch[ind] * X_batch[ind])
            dw += di
            
        dw = dw/len(Y_batch)
        return dw
    
    def SGD(self, features, outputs):
        from sklearn.utils import shuffle
        max_epochs = 5000
        weights = np.zeros(features.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01
        
        for epoch in range(1, max_epochs):
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                ascent = self.GradientCost(weights, x, Y[ind])
                weights = weights - (self.learning_rate * ascent)
            
            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.Cost(weights, features, outputs)
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
                nth += 1
        return weights
    

