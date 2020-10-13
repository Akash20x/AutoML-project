import pandas as pd
import numpy as np
import random

class decisiontree():
    
    def __init__(self, counter, min_samples, max_depth, random_subspace):
        self.counter = counter
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.random_subspace = random_subspace

    def determine_type_of_feature(self, df):
        
        feature_types = []
        n_unique_values_treshold = 15
        for feature in df.columns:
            if feature != "label":
                unique_values = df[feature].unique()
                example_value = unique_values[0]
    
                if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                    feature_types.append("categorical")
                else:
                    feature_types.append("continuous")
        
        return feature_types
    
    def check_purity(self, data):
        
        label_column = data[:, -1]
        unique_classes = np.unique(label_column)
    
        if len(unique_classes) == 1:
            return True
        else:
            return False
    
        
    # 1.2 Classify
    def classify_data(self, data):
        
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]
        
        return classification
    
    
    # 1.3 Potential splits?
    def get_potential_splits(self, data):
        
        potential_splits = {}
        _, n_columns = data.shape
        column_indices = list(range(n_columns - 1))    # excluding the last column which is the label
        
        if self.random_subspace and self.random_subspace <= len(column_indices):
            column_indices = random.sample(population=column_indices, k=self.random_subspace)
        
        for column_index in column_indices:          
            values = data[:, column_index]
            unique_values = np.unique(values)
            
            potential_splits[column_index] = unique_values
        
        return potential_splits
    
    
    # 1.4 Lowest Overall Entropy?
    def calculate_entropy(self, data):
        
        label_column = data[:, -1]
        _, counts = np.unique(label_column, return_counts=True)
    
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
         
        return entropy
    
    
    def calculate_overall_entropy(self, data_below, data_above):
        
        n = len(data_below) + len(data_above)
        p_data_below = len(data_below) / n
        p_data_above = len(data_above) / n
    
        overall_entropy =  (p_data_below * self.calculate_entropy(data_below) 
                          + p_data_above * self.calculate_entropy(data_above))
        
        return overall_entropy
    
    
    def determine_best_split(self, data, potential_splits):
        
        overall_entropy = 9999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(data, split_column=column_index, split_value=value)
                current_overall_entropy = self.calculate_overall_entropy(data_below, data_above)
                
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
        
        return best_split_column, best_split_value
    
    
    # 1.5 Split data
    def split_data(self, data, split_column, split_value):
        
        split_column_values = data[:, split_column]
    
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            data_below = data[split_column_values <= split_value]
            data_above = data[split_column_values >  split_value]
        
        # feature is categorical   
        else:
            data_below = data[split_column_values == split_value]
            data_above = data[split_column_values != split_value]
        
        return data_below, data_above
    
    
    # 2. Decision Tree Algorithm
    def decision_tree_algorithm(self, df):
        
        # data preparations
        if self.counter == 0:
            global COLUMN_HEADERS, FEATURE_TYPES
            COLUMN_HEADERS = df.columns
            FEATURE_TYPES = self.determine_type_of_feature(df)
            data = df.values
        else:
            data = df           
        
        
        # base cases
        if (self.check_purity(data)) or (len(data) < self.min_samples) or (self.counter == self.max_depth):
            classification = self.classify_data(data)
            
            return classification
    
        
        # recursive part
        else:    
            self.counter += 1
    
            # helper functions 
            potential_splits = self.get_potential_splits(data)
            split_column, split_value = self.determine_best_split(data, potential_splits)
            data_below, data_above = self.split_data(data, split_column, split_value)
            
            # check for empty data
            if len(data_below) == 0 or len(data_above) == 0:
                classification = self.classify_data(data)
                return classification
            
            # determine question
            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)
                
            # feature is categorical
            else:
                question = "{} = {}".format(feature_name, split_value)
            
            # instantiate sub-tree
            sub_tree = {question: []}
            
            # find answers (recursion)
            yes_answer = self.decision_tree_algorithm(data_below)
            no_answer = self.decision_tree_algorithm(data_above)
            
            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (self.min_samples or self.max_depth base case).
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
            
            return sub_tree
    
        
    # 3.1 One example
    def predict_example(self, example, tree):
        
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value, = question.split(" ")
    
        # ask question
        if comparison_operator == "<=":
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        
        # feature is categorical
        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]
    
        # base case
        if not isinstance(answer, dict):
            return answer
        
        # recursive part
        else:
            residual_tree = answer
            return self.predict_example(example, residual_tree)
    
    
    def decision_tree_predictions(self, test_df, tree):
        predictions = test_df.apply(self.predict_example, args=(tree,), axis=1)
        return predictions
    
