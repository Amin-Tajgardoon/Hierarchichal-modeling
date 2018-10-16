# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:57:30 2018

@author: Amin
"""

import os
import sys
lib_path = os.path.abspath(os.path.join('__file__', '..', '..'))
print(lib_path)
sys.path.append(lib_path)
#sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
 
import ML.utils as utils

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import time
import pickle


random_state = 0
np.random.seed(random_state)

outdir = '../../output/Experiment 4'

# ### read targets
targets = pd.read_csv("../../data/targets.csv", index_col=0)

# ### select targets which have more than half non-null samples and have at-least 10 positive samples
notna_sum = targets.notna().sum()
targets = targets.loc[:, (notna_sum > targets.shape[0]/2) & (targets.sum(axis=0) >= 10)]

# ### read all features with physician features and without physician info
features_w_phys = pd.read_csv("../../data/all_features_178_patients_with_physicians.csv", index_col=0)
features = pd.read_csv("../../data/all_features_178_patients.csv", index_col=0)
datasets = {'with_phys': features_w_phys, 'no_phys':features}

# ### model- selection with  cross-validation
# at each iteration:
#         - impute missing values
#         - ANOVA feature selection
#           

models = ['LR-L2', 'LR-L1', 'RF']

predictions = {key:{} for key in models}
for k in predictions.keys():
    predictions[k] = {key:{} for key in targets.columns}
    for k2 in targets.columns:
        predictions[k][k2] = {key:[] for key in datasets.keys()}
        
start_time = time.time()

for target in targets.columns:
    print('target:', target)
    
    y = targets.loc[:, target].copy()
    mask = y.notna().values
    y = y.loc[mask]

    for data_key in datasets.keys():
        
        X = datasets[data_key].loc[mask, :].copy()
       
        skf = StratifiedKFold(n_splits=3, random_state=random_state)
        
        for model in models:
            predictions[model][target][data_key] = np.zeros(y.size)
            
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            loop_start_time = time.time()
            #print('iteration', test_idx+1, 'of', len(rows_idx))
        
            ## drop all-null columns from train and transform to test_instance
            X_train.dropna(axis=1, how='all', inplace=True)
            X_test = X_test[X_train.columns]
            
            ## first drop constant columns to speedup feature-selection
            X_train = X_train.loc[:, X_train.nunique() != 1]
            X_test = X_test[X_train.columns]
    
    
            ## impute missing values with mean and trasform to test_instance
            col_names = X_train.columns
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
    
            ## normalize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            ## re-assign colnames to train and test (convert back to dataframe and series)
            X_train = pd.DataFrame(X_train, columns = col_names)
            X_test = pd.DataFrame(X_test, columns=col_names)
            
    
            ## feature-selection for each target columns
            selected_features = utils.anova_feature_selection_2(X_train, y_train, pval_threshold=0.01)
    
            print('#selected features=', selected_features.size)
    
            ## reduce train and test columns to selected_features
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
    
            for model in models:
                print('model=',model)
                

                if model == 'LR-L2':    
                    clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state= random_state)
                elif model == 'LR-L1':
                    clf = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state= random_state)
                elif model == 'RF':
                    clf = RandomForestClassifier(n_estimators=15, random_state= random_state)
                else:
                    raise "unsupported model:" + model

                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_test)[:,1]
                predictions[model][target][data_key][test_index] = probs

            print('iteration_time=', time.time() - loop_start_time)
            with open(outdir + '/predictions.pkl', 'wb') as f:
                pickle.dump(predictions, f)
            #utils.write_predictions_2(predictions, outdir + '/predictions.csv')
utils.write_eval_results_2(utils.evaluate_2(targets, predictions), outdir + '/eval_results.csv')
    
total_time = time.time() - start_time
print('total time=', total_time)
