
import pandas as pd
import numpy as np

class Preprocessing:
    
    def __init__(self, df, target, scaletype):
        self.df = df
        self.target = target
        self.scaletype = scaletype
    
    def MissingValImp(self):
        global cols
        cols = self.df.columns
        global df_num 
        df_num = self.df._get_numeric_data()
        global df_cat 
        df_cat = list(set(self.df.columns) - set(df_num))
        for col in df_num.columns:
            if df_num[col].isnull().sum() > 0:
                df_num[str(col) + '_mean'] = df_num[col].fillna(df_num[col].mean())
                df_num[str(col)] = df_num[col].fillna(df_num[col].median())
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='most_frequent')
        self.df = pd.DataFrame(imp.fit_transform(self.df), columns=cols)
        return self.df
    
    def Feature_removal(self):
        from sklearn.feature_selection import VarianceThreshold
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(self.df)
        self.df = constant_filter.transform(self.df)
        self.df = pd.DataFrame(self.df)
        return self.df
    
    def TargetEncoding(self):
        Y = self.df[self.target]
        if Y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            Y = le.fit_transform(Y)
            Y = pd.DataFrame(Y)
            Y.columns = [self.target]
        else:
            Y = Y   
        return Y

    def SimpleCatEncoding(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.df = self.df.apply(LabelEncoder().fit_transform)
        return self.df
    
    def Scaling(self):
        from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
        if self.scaletype == 'MinMaxScaler':
            self.df = MinMaxScaler().fit_transform(self.df)
            self.df = pd.DataFrame(self.df)
        if self.scaletype == 'StandardScaler':
            self.df = StandardScaler().fit_transform(self.df)
            self.df = pd.DataFrame(self.df)
        if self.scaletype == 'QuantileTransformer':
            self.df = QuantileTransformer().fit_transform(self.df)
            self.df = pd.DataFrame(self.df)
        return self.df
    
    def Outliers(self):
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators = 50, max_samples = 'auto', contamination = float(0.1))
        model.fit(self.df)
        scores=model.decision_function(self.df)
        self.df['anomalies'] = model.predict(self.df)
        condition = (self.df['anomalies'] == 1)
        self.df = self.df[condition]
        self.df = self.df.drop(['anomalies'], axis = 1)
        return self.df
    
    def Correlaton_selection(self):
        corr = self.df.corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        selected_columns = self.df.columns[columns]
        self.df = self.df[selected_columns]
        return self.df
    
    def RemoveEmptyColumns(self):
        for col in self.df.columns:
            if (self.df[col].isnull().sum() == self.df.shape[0]):
                self.df = self.df.drop([col], axis = 1)
        return self.df
    
    def removecolumnspace(self):
        self.df.columns = self.df.columns.str.replace(' ', '_')
        return self.df
    

