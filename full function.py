'''
Further jobs:
Add other models
Hyperparameter tuning
Add AIC, other metrics ,such as auc.
'''


# Setting working directory for test datasets
import os
os.chdir("C:\\Users\\jacob\\OneDrive\\Documents\\Infovio Data Science Internship\\autoML\\Test Datasets")

# Appending additional folder where models are stored
import sys
sys.path.append("C:\\Users\\jacob\\OneDrive\\Documents\\Infovio Data Science Internship\\autoML full")

# Importing models
from preprocessingoop import Preprocessing
from decision_trees import decisiontree
from random_forest import randomforest
from helperfunctions import HelperFunctions
from svm import SVM
from principalcomponentanalysis import PCA
import pandas as pd
import numpy as np
from knn import KNN
from hyperopt.pyll.stochastic import sample
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials



#path = C:\\Users\\jacob\\OneDrive\\Documents\\Infovio Data Science Internship\\autoML\\Test Datasets\\titanictrain.csv

def main():
    
    ## User input
    
    data_path = input("Enter the path to your input file. For example [C:\\Users\\User\\Documents\\Datasets\\data.csv]:")
    
    problemtype = input("What is your ML problem type; Enter [c] for classification, [r] for regression:")
    
    df = pd.read_csv(data_path)
    
    target = input("Enter the target variable column name for your dataset:")
    
    scaletype = input("Enter the scaling type for the dataset, options include [MinMaxScaler], [QuantileTransformer], [StandardScaler]:")
    
    #Creating an empty list to store accuracies
    results = []
    
    ## Preprocessing data
    
    print("                                ")
    print("---------------------------------")
    print("        Cleaning the data        ")
    print("---------------------------------")
    print("                                ")
    
    pre = Preprocessing(df, target, scaletype)
    targettype = df[target].dtype
    print("Encoding the target variable if it is categorical...")
    targetY = pre.TargetEncoding()
    print("Removing any empty columns from the dataset...")
    df = pre.RemoveEmptyColumns()
    print("Imputing missing values into the dataset...")
    df = pre.MissingValImp()
    print("Encoding categorical variables...")
    df = pre.SimpleCatEncoding()
    columns = df.columns
    print("Scaling the values...")
    df = pre.Scaling()
    df.columns = columns
    print("Selecting the most important features based on correlation")
    df = pre.Correlaton_selection()
    df = pd.concat([df, targetY], axis = 1)
    print("Checking the data for any outliers and removing them...")
    df = pre.Outliers()
    df['label'] = df[target]
    df = df.drop([target], axis = 1)
    df = pre.removecolumnspace()
    df = df.drop([target], axis = 1)
    print("Data cleaning finished")
    
    ## Option to perform pca for large data
    
    
    print("                                 ")
    print("---------------------------------")
    print("               PCA               ")
    print("---------------------------------")
    print("                                 ")
    
    yn = input("Would you like to perform principal component analysis on the dataset? Enter [y] for yes, [n] for no:")
    
    if yn == 'y':
        label = df['label']
        n_components = int(input("How many components would you like to reduce the dataset to?:"))
        pca = PCA(n_components)
        pca.fit(df)
        df = pca.transform(df)
        df = pd.DataFrame(df)
        df['label'] = label
    
    ## Train Test split
    print("                                ")
    print("---------------------------------")
    print("Splitting the data into train and test sets")
    print("---------------------------------")
    print("                                ")
    
    test_size = float(input("Input the test size for the split of test and train sets:"))
    hf = HelperFunctions(test_size)
    global train,test
    train, test = hf.train_test_split(df)
    global X_train
    X_train = train.drop(['label'], axis = 1)
    global X_test
    X_test = test.drop(['label'], axis = 1)
    global y_train
    y_train = train['label']
    global y_test
    y_test = test['label']
    
    print("                                ")
    print("---------------------------------")
    print("         Training models         ")
    print("---------------------------------")
    print("                                ")
    
    print("                                ")
    print("---------------------------------")
    print("          Decision Tree          ")
    print("---------------------------------")
    print("                                ")
    
    print("Optimizing parameters...")
    
    if problemtype == 'c':

        def hpdt(params):
            dt = decisiontree(**params)
            tree = dt.decision_tree_algorithm(train)
            dtpredictions = dt.decision_tree_predictions(test, tree)
            dtaccuracy = hf.calculate_accuracy(dtpredictions, y_test)
            return dtaccuracy
    
        spacedt = {
            'counter': hp.choice('counter', [0]),
            'min_samples': hp.choice('min_samples', range(1,5)),
            'max_depth': hp.choice('max_depth', range(1,20)),
            'random_subspace': hp.choice('random_subspace', [None])
            }
        
        def fdt(params):
            acc = hpdt(params)
            return {'loss': -acc, 'status': STATUS_OK}
        
        trials = Trials()
        max_evals = int(input("Enter your value for the maximum number of evaluations you want. ():"))
        best = fmin(fdt, spacedt, algo = tpe.suggest, max_evals = max_evals, trials = trials)
        
        counter = best.get('counter')
        max_depth = best.get('max_depth')
        min_samples = best.get('min_samples')
        random_subspace = best.get('random_subspace')
    
    else: 
        def hpdt(params):
            dt = decisiontree(**params)
            tree = dt.decision_tree_algorithm(train)
            dtpredictions = dt.decision_tree_predictions(test, tree)
            dtaccuracy = hf.rmse(dtpredictions, y_test)
            return dtaccuracy
        
        spacedt = {
        'counter': hp.choice('counter', [0]),
        'min_samples': hp.choice('min_samples', range(1,5)),
        'max_depth': hp.choice('max_depth', range(1,20)),
        'random_subspace': hp.choice('random_subspace', [None])
        }
        
        def fdt(params):
            acc = hpdt(params)
            return {'loss': acc, 'status': STATUS_OK}
    
        trials = Trials()
        max_evals = int(input("Enter your value for the maximum number of evaluations you want:"))
        best = fmin(fdt, spacedt, algo = tpe.suggest, max_evals = max_evals, trials = trials)
        
        counter = best.get('counter')
        max_depth = best.get('max_depth')
        min_samples = best.get('min_samples')
        random_subspace = best.get('random_subspace')
        
    print('Optimal hyperparameters for decision tree are: counter = {}, max_depth = {}, min_samples = {}, random_subspace = {}.'.format(counter, max_depth, min_samples, random_subspace))
    
    dt = decisiontree(counter=counter, min_samples=min_samples, max_depth=max_depth, random_subspace=random_subspace)
    tree = dt.decision_tree_algorithm(train)
    print("Predicting values...")
    dtpredictions = dt.decision_tree_predictions(test, tree)
    if problemtype == "c":
        print("Calculating Accuracy...")
        dtaccuracy = hf.calculate_accuracy(dtpredictions, test['label'])
        print("Accuracy of decision tree predictions = {:.2f}".format(dtaccuracy*100)+'%')
        results.append(dtaccuracy)
    else:
        print("Calculating RMSE...")
        dtrootmean = hf.rmse(dtpredictions, test['label'])
        print("RMSE of decision tree predictions = {}".format(dtrootmean))
        results.append(dtrootmean)
    
    print("                                ")
    print("---------------------------------")
    print("          Random Forest          ")
    print("---------------------------------")
    print("                                ")
    
    print("Optimizing parameters...")
    
    if problemtype == 'c':

        def hpdt(params):
            rf = randomforest(**params)
            forest = rf.fit(train)
            for i in range(0, len(forest)-1):
                if type(forest[i]) == np.float64:
                    del forest[i]
            rfpredictions = rf.predict(test, forest, 'c')
            rfaccuracy = hf.calculate_accuracy(rfpredictions, test['label'])
            return rfaccuracy
    
        spacerf = {
            'n_bootstrap': hp.choice('n_bootstrap', range(50, len(train))),
            'n_trees': hp.choice('n_trees', range(1, 20)),
            'n_features': hp.choice('n_features', range(1, 5)),
            'dt_max_depth': hp.choice('dt_max_depth', range(1, 20)),
            'min_samples': hp.choice('min_samples', range(1, 5))
            }
        
        def frf(params):
            acc = hpdt(params)
            return {'loss': -acc, 'status': STATUS_OK}
        
        trials = Trials()
        max_evals = int(input("Enter your value for the maximum number of evaluations you want:"))
        best = fmin(frf, spacerf, algo = tpe.suggest, max_evals = max_evals, trials = trials)
    
    else:
        def hpdt(params):
            rf = randomforest(**params)
            forest = rf.fit(train)
            for i in range(0, len(forest)-1):
                if type(forest[i]) == np.float64:
                    del forest[i]
            rfpredictions = rf.predict(test, forest, 'r')
            rfaccuracy = hf.rmse(rfpredictions, test['label'])
            return rfaccuracy
    
        spacerf = {
            'n_bootstrap': hp.choice('n_bootstrap', range(50, len(train))),
            'n_trees': hp.choice('n_trees', range(1, 20)),
            'n_features': hp.choice('n_features', range(1, 5)),
            'dt_max_depth': hp.choice('dt_max_depth', range(1, 20)),
            'min_samples': hp.choice('min_samples', range(1, 5))
            }
        
        def frf(params):
            acc = hpdt(params)
            return {'loss': acc, 'status': STATUS_OK}
        
        trials = Trials()
        max_evals = int(input("Enter your value for the maximum number of evaluations you want:"))
        best = fmin(frf, spacerf, algo = tpe.suggest, max_evals = max_evals, trials = trials)
    
    
    n_bootstrap = best.get('n_bootstrap')
    n_trees = best.get('n_trees')
    n_features = best.get('n_features')
    dt_max_depth = best.get('dt_max_depth')
    min_samples = best.get('min_samples')
        
    print('Optimal hyperparameters for random forest are: n_bootstrap = {}, n_trees = {}, n_features = {}, dt_max_depth = {}, min_samples = {}.'.format(n_bootstrap, n_trees, n_features, dt_max_depth, min_samples))
    
    print("Building random forest...")
    rf = randomforest(n_bootstrap = n_bootstrap, n_trees = n_trees, n_features = n_features, dt_max_depth = dt_max_depth, min_samples = min_samples)
    forest = rf.fit(train)
    for i in range(0, len(forest)-1):
            if type(forest[i]) == np.float64:
                del forest[i]
    print("Predicting values...")
    rfpredictions = rf.predict(test, forest, problemtype)
    if problemtype == "c":
        print("Calculating Accuracy...")
        rfaccuracy = hf.calculate_accuracy(rfpredictions, test['label'])
        print("Accuracy of random forest predictions = {:.2f}".format(rfaccuracy*100)+'%')
        results.append(rfaccuracy)
    else:
        print("Calculating RMSE...")
        rfrootmean = hf.rmse(rfpredictions, test['label'])
        print("RMSE of random forest predictions = {}".format(rfrootmean))
        results.append(rfrootmean)
        
    print("                                ")
    print("---------------------------------")
    print("     Support Vector Machines     ")
    print("---------------------------------")
    print("                                ")
    
    # Hyperparameter optimization
    
    if problemtype == 'c':
        svmdf = df
        svmdf.insert(loc = len(df.columns), column= 'intercept', value = 1)
        train, test = hf.train_test_split(svmdf)
        X_train = train.drop(['label'], axis = 1)
        X_test = test.drop(['label'], axis = 1)
        y_train = train['label']
        y_test = test['label']
        print("Optimizing hyperparameters...")
        def hpsvm(params):
            svm = SVM(**params)
            W = svm.SGD(X_train.to_numpy(), y_train.to_numpy())
            y_test_predicted = np.array([])
            y_train_predicted = np.array([])
            for i in range(X_train.shape[0]):
                yp = np.sign(np.dot(X_train.to_numpy()[i], W))
                y_train_predicted = np.append(y_train_predicted, yp)
            for i in range(X_test.shape[0]):
                yp = np.sign(np.dot(X_test.to_numpy()[i], W))
                y_test_predicted = np.append(y_test_predicted, yp)
            svmaccuracy = hf.calculate_accuracy(y_test.to_numpy(), y_test_predicted)
            return svmaccuracy
    
        spacesvm = {
            'reg_strength': hp.uniform('reg_strength', 100, 10000),
            'learning_rate': hp.uniform('learning_rate', 0.00001, 0.0001)
            }
        
        def f(params):
            acc = hpsvm(params)
            return {'loss': -acc, 'status': STATUS_OK}
        
        max_evals = int(input("Enter your value for the maximum number of evaluations you want:"))
        
        trials = Trials()
        best = fmin(f, spacesvm, algo = tpe.suggest, max_evals = max_evals, trials = trials)
        
        learning_rate = best.get('learning_rate')
        reg_strength = best.get('reg_strength')
        
        print('Optimal hyperparameters for SVM are: learning rate = {}, regression strength = {}'.format(learning_rate, reg_strength))
        
        svm = SVM(reg_strength, learning_rate)
        print("training started...")
        W = svm.SGD(X_train.to_numpy(), y_train.to_numpy())
        print("training finished")
        y_test_predicted = np.array([])
        print("testing the model...")
        y_train_predicted = np.array([])
        for i in range(X_train.shape[0]):
            yp = np.sign(np.dot(X_train.to_numpy()[i], W))
            y_train_predicted = np.append(y_train_predicted, yp)
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(X_test.to_numpy()[i], W))
            y_test_predicted = np.append(y_test_predicted, yp)
        svmaccuracy = hf.calculate_accuracy(y_test.to_numpy(), y_test_predicted)
        print("accuracy on test dataset: {:.2f}".format(svmaccuracy*100)+'%')
        results.append(svmaccuracy)
    else:
        print("SVM cannot be used for regression.")
    
        
    
    print("                                ")
    print("---------------------------------")
    print("       K nearest neighbour       ")
    print("---------------------------------")
    print("                                ")
    
    if problemtype == "c":
    
        X_trainknn = X_train.to_numpy()
        X_testknn = X_test.to_numpy()
        y_trainknn = y_train.to_numpy()
        y_testknn = y_test.to_numpy()
        
        def hpknn(params):
            clf = KNN(**params)
            clf.fit(X_trainknn, y_trainknn)
            predictions = clf.predict(X_testknn)
            accuracy = hf.calculate_accuracy(y_testknn, predictions)
            return accuracy
    
        spaceknn = {
            'k': hp.choice('k', range(1, 100))
        }
        
        def f(params):
            acc = hpknn(params)
            return {'loss': -acc, 'status': STATUS_OK}
        
        trials = Trials()
        
        # Finding best hyperparameters
        
        print("Optimizing hyperparameters...")
        
        max_evals = int(input("Enter your value for the maximum number of evaluations you want:"))
    
        best = fmin(f, spaceknn, algo = tpe.suggest, max_evals = max_evals, trials = trials)
        
        k = best.get('k')
        
        print('Optimal hyperparameters for KNN are: k = {}'.format(k))
        
    
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        print("Training started...")
        clf = KNN(k=int(k))
        clf.fit(X_train, y_train)
        print("Training finished")
        print("Predicting values...")
        predictions = clf.predict(X_test)
        knnaccuracy = hf.calculate_accuracy(y_test, predictions)
        print("Accuracy of knn predictions are {:.2f}".format(knnaccuracy*100)+'%')
        results.append(knnaccuracy)

    else:
        pass
    
    ## Model Decision
    print("                                ")
    if problemtype == 'c':
        print("                                ")
        print("----------------------------")
        print("     Accuracy of models     ")
        print("----------------------------")
        print("                                ")
        print('Decision Tree      '+"{:.2f}".format(dtaccuracy*100)+'%')
        print('Random Forest      '+"{:.2f}".format(rfaccuracy*100)+'%')
        print('SVM                '+"{:.2f}".format(svmaccuracy*100)+'%')
        print('KNN                '+"{:.2f}".format(knnaccuracy*100)+'%')
    else:
        print("                                ")
        print("----------------------------")
        print("     RMSE of models     ")
        print("----------------------------")
        print("                                ")
        print('Decision Tree      '+"{:.2f}".format(dtrootmean))
        print('Random Forest      '+"{:.2f}".format(rfrootmean))
    
    print("                                ")
    
    if problemtype == 'c':
        print("Maximum accuracy score is {}".format(max(results)*100)+'%')
    else:
        print("Minimum rmse is {}".format(min(results)))

## Creating a dataframe with 3 columns then outputting max accruacy/minrms
results = pd.DataFrame([], columns = ['Model name', 'Accuracy/RMSE', 'Predictions'])