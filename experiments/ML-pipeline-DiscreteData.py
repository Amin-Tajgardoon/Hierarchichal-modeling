import os
import sys
lib_path = os.path.abspath(os.path.join('__file__', '..', '..'))
print(lib_path)
sys.path.append(lib_path)
#sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
 
import ML.utils as utils
from ML.utils import make_names_R_compatible

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, roc_auc_score
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss

from functools import reduce

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import time


random_state = 0
np.random.seed(random_state)
pandas2ri.activate()
outdir = '../../output/Experiment 3'

# ### initialize target groups and make names R-compatible

target_grps = [#['K', 'CL', 'CO2', 'NA'],
               #['CREAT', 'BUN'],
               #['INR', 'PT']]#,
               ['CA', 'CL', 'MG', 'LACT']]

### define a union list of all targets
all_labels = []
for grp in target_grps:
    all_labels += [l for l in grp]
all_labels = np.sort(np.unique(np.array(all_labels))).tolist()

# ### read targets
targets = pd.read_csv("../../data/targets.csv", index_col=0).loc[:, all_labels]

# ### read all features
features = pd.read_csv("../../data/all_features_178_patients.csv", index_col=0)


# ### make columns names compatible with R
# ### drop duplicated colnames
features.columns = make_names_R_compatible(features.columns)
features = features.loc[:, ~features.columns.duplicated(keep='first')]
targets.columns = make_names_R_compatible(targets.columns)
for i in range(0, len(target_grps)):
    target_grps[i] = make_names_R_compatible(target_grps[i])
all_labels = []
for grp in target_grps:
    all_labels += [l for l in grp]
all_labels = np.sort(np.unique(np.array(all_labels))).tolist()


# ### model- selection with leave-one-out cross-validation
# at each iteration:
#         - impute missing values
#           - discretize variables
#         - ANOVA feature selection
#           

models = ['LR-L2', 'LR-L1', 'RF', 'BN']

start_time = time.time()

for target_grp in target_grps:
    print('targets:', target_grp)
    predictions = {key:{} for key in models}
    for k in predictions.keys():
        predictions[k] = {key:[] for key in target_grp}

    rows_idx = range(0,features.shape[0])

    for test_idx in rows_idx:
        loop_start_time = time.time()
        print('iteration', test_idx+1, 'of', len(rows_idx))

        train_idx = [i for i in range(0, features.shape[0]) if i != test_idx]
        train_set = features.iloc[train_idx, :].copy()
        test_instance = features.iloc[test_idx, :].copy()

        ## drop all-null columns from train and transform to test_instance
        train_set.dropna(axis=1, how='all', inplace=True)
        test_instance = test_instance[train_set.columns]
        
        ## first drop constant columns to speedup feature-selection
        train_set = train_set.loc[:, train_set.nunique() != 1]
        test_instance = test_instance[train_set.columns]


        ## impute missing values with mean and trasform to test_instance
        col_names = train_set.columns
        imputer = SimpleImputer(strategy='mean')
        train_set = imputer.fit_transform(train_set)
        test_instance = imputer.transform(test_instance.values.reshape(1, -1))

        ## normalize features
        scaler = StandardScaler()
        train_set = scaler.fit_transform(train_set)
        test_instance = scaler.transform(test_instance)
        
        ## discretize features
        disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
        train_set = disc.fit_transform(train_set)
        test_instance = disc.transform(test_instance)
        
        ## re-assign colnames to train and test (convert back to dataframe and series)
        train_set = pd.DataFrame(train_set, columns = col_names)
        test_instance = pd.Series(test_instance[0], index=col_names)
        
        ## second drop constant columns to speedup feature-selection
        train_set = train_set.loc[:, train_set.nunique() != 1]
        test_instance = test_instance[train_set.columns]

        ## feature-selection for each target columns
        selected_features = utils.anova_feature_selection(train_set, targets[target_grp], pval_threshold=0.01)

        ## union selected features for all targets
        union_selected_features = reduce(np.union1d, tuple([feature_list for _, feature_list in selected_features.items()]))

        print('#selected features=', union_selected_features.size)

        ## reduce train and test columns to union_selected_features
        train_set = train_set[union_selected_features]
        test_instance = test_instance[union_selected_features]

        ## create blacklist as prior knowledge for BN model
        blacklist = utils.create_BN_blacklist(feature_names=train_set.columns.tolist(), target_names=target_grp)       

        for model in models:
            print('model=',model)

            ## train and test BN model
            if model == 'BN':
                ## select rows for which all targets are available
                ## reset indices of X and y to allow join on index
                mask = targets[target_grp].notna().all(axis=1).values
                y = targets[target_grp].loc[mask, :].copy().reset_index(drop=True)
                X = train_set.loc[mask, :].copy().reset_index(drop=True)
                ## join X and y for BN algorithms
                X = X.join(y)

                ## augment test_instance with target variables for BN algorithms
                joined_test_instance = test_instance.append(targets[target_grp].iloc[test_idx, :])

                ## fit BN model
                fitted = utils.fit_bn_disc(X, blacklist, score = 'bde', method='bayes', restart=3, maxp=10)

                ## predict for each target
                for target_name in target_grp:
                    predictions[model][target_name].append(utils.bn_predict_disc(fitted, target_name, target_grp, joined_test_instance))

            ## train and test other models (one for each target)
            else:
                for target_name in target_grp:
                    ## select rows where target is not null
                    mask = targets[target_name].notna().values
                    y = targets.loc[mask, target_name].copy()
                    X = train_set.loc[mask, :].copy()

                    if model == 'LR-L2':    
                        clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state= random_state)
                    elif model == 'LR-L1':
                        clf = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state= random_state)
                    elif model == 'RF':
                        clf = RandomForestClassifier(n_estimators=15, random_state= random_state)
                    else:
                        raise "unsupported model:" + model

                    clf.fit(X, y)
                    predictions[model][target_name].append(clf.predict_proba(test_instance.values.reshape(1,-1))[0][1])

        print('iteration_time=', time.time() - loop_start_time)
        utils.write_predictions(predictions, outdir + '/predictions-' + '_'.join(target_grp) + '.csv')
    utils.write_eval_results(utils.evaluate(targets[target_grp], predictions), outdir + '/eval_results-' + '_'.join(target_grp) + '.csv')

total_time = time.time() - start_time
print('total time=', total_time)
