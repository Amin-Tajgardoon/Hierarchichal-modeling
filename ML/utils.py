import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score, roc_auc_score, coverage_error, label_ranking_average_precision_score, label_ranking_loss

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

R_libpath = 'C:/Users/Amin/Documents/R/win-library/3.5'
def anova_feature_selection(features, targets, pval_threshold = 0.05, debug=False):
    selected_features = {}
    for label in targets.columns:

        ## select rows where target is not null
        mask = targets[label].notna().values
        y = targets.loc[mask, label]
        X = features.loc[mask, :].copy()

        ## ANOVA feature-selection
        f, pval = f_classif(X, y)
        f = pd.Series(f).replace([np.inf, -np.inf], np.nan)
        pval[f.isna()] = np.nan

        ## select features with p-val< pval_threshold
        selected_features[label] = X.columns[pval < pval_threshold]

        if debug:
            print('target=', label)
            print('features dimension=', X.shape)
            print('#selected features=', selected_features[label].size)
            print('#'*40)
    return selected_features


def anova_feature_selection_2(X, y, pval_threshold = 0.05, debug=False):

    ## ANOVA feature-selection
    f, pval = f_classif(X, y)
    f = pd.Series(f).replace([np.inf, -np.inf], np.nan)
    pval[f.isna()] = np.nan

    ## select features with p-val< pval_threshold
    selected_features = X.columns[pval < pval_threshold]

    return selected_features


def create_BN_blacklist(feature_names, target_names):
    '''
    feature_names: list of feature names
    target_names: list of target names
    
    returns: blacklist dataframe for bnlearn algorithms
    '''
    
    forbidden_targets2features = pd.DataFrame({'from':[], 'to':[]})
    for name in target_names:
        forbidden_edges = pd.DataFrame({'from': np.repeat(name, len(feature_names)), 'to': feature_names})
        forbidden_targets2features = pd.concat([forbidden_targets2features, forbidden_edges], ignore_index=True)
    
    forbidden_among_features = pd.DataFrame({'from':[], 'to':[]})
    for name in feature_names:
        forbidden_edges = pd.DataFrame({'from': np.repeat(name, len(feature_names)), 'to': feature_names})
        forbidden_among_features = pd.concat([forbidden_among_features, forbidden_edges], ignore_index=True)
        
    blacklist = pd.concat([forbidden_among_features, forbidden_targets2features], ignore_index=True)
    
    return blacklist


def fit_bn_disc(data, blacklist, score, method, restart=3, maxp=10):
    bnlearn = importr('bnlearn', lib_loc = R_libpath)
    base = importr('base', lib_loc = R_libpath)

    r_data = pandas2ri.py2ri(data)
    r_data = base.as_data_frame(base.sapply(r_data, base.ordered))
    r_blacklist = pandas2ri.py2ri(blacklist)
    r_bn_struct = bnlearn.hc(x=r_data, balcklist=r_blacklist, score=score, restart=restart, maxp=maxp)
    r_fitted = bnlearn.bn_fit(x=r_bn_struct, data=r_data, method=method)
    return r_fitted


def bn_predict_disc(r_fitted, target_name, target_names, test_instance):
    '''
    use bnlearn to obtain the conditional probability of target in discrete Bayesian networks
    condition set is the markov blanket of the target node
    '''
    bnlearn = importr('bnlearn', lib_loc=R_libpath)
    robjects.globalenv['fitted'] = r_fitted
    
    nonTargets_mask = ~test_instance.index.isin(target_names)
    markovBlanket_mask = test_instance.index.isin(list(bnlearn.mb(r_fitted, target_name)))
    mask = (nonTargets_mask & markovBlanket_mask)
    
    evidence = ' & '.join(["(" + name + "==" + str(value) + ")" for name, value in zip(test_instance.index.values[mask], test_instance.values[mask])])
    event = target_name + " == 1"
    
    prob_1 = robjects.r('cpquery(fitted, event = ' + event + ', evidence= (' + evidence + '))')[0]
    
    return prob_1


def predictionsDict2df(predictions):
    df = pd.DataFrame.from_dict({(i,j): predictions[i][j] 
                                for i in predictions.keys() 
                                for j in predictions[i].keys()},
                                orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)

    return df


def write_predictions(predictions, outpath):
    df = predictionsDict2df(predictions)
    df.to_csv(outpath)

def predictionsDict2df_2(predictions):
    '''
    predictions is a multiindex dict with 3 index levels: (model, target, dataset)
    '''
    df = pd.DataFrame.from_dict({(i,j,k): predictions[i][j][k] 
                                for i in predictions.keys() 
                                for j in predictions[i].keys()
                                for k in predictions[i][j].keys()},
                                orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)

    return df


def write_predictions_2(predictions, outpath):
    df = predictionsDict2df_2(predictions)
    df.to_csv(outpath)


def evaluate(targets, predictions):
    predictions = predictionsDict2df(predictions)
    results = {}
    replace_null_probs_with = 0

    averaging_methods = ['micro', 'macro', 'weighted'] #, 'samples']

    mask = targets.notna().all(axis=1).values
    y_true = targets.loc[mask, :]

    for model in predictions.index.levels[0]:

        y_score = predictions.loc[model].loc[y_true.columns, mask].T
        num_nulls = y_score.isnull().sum().sum()
        if (num_nulls > 0):
            print('WARNING: prediction scores for model {0} contain {1} null values. replaced with {2}'.format(model, num_nulls, replace_null_probs_with))
            y_score.fillna(replace_null_probs_with, inplace=True)

        results[(model, 'coverage_error')] = coverage_error(y_true, y_score)
        results[(model, 'ranking_avg_prec')] = label_ranking_average_precision_score(y_true, y_score)
        results[(model, 'ranking_loss')] = label_ranking_loss(y_true, y_score)

        for avrg_method in averaging_methods:
            results[(model, 'avg_prec-' + avrg_method)] = average_precision_score(y_true, y_score, average=avrg_method)
            results[(model, 'avg_auroc-' + avrg_method)] = roc_auc_score(y_true, y_score, average=avrg_method)

        single_target_auroc = roc_auc_score(y_true, y_score, average=None)
        for target, auroc in zip(y_true.columns, single_target_auroc):
            results[(model, 'auroc-' + target)] = auroc

        single_target_auprc = average_precision_score(y_true, y_score, average=None)
        for target, auprc in zip(y_true.columns, single_target_auprc):
            results[(model, 'auprc-' + target)] = auprc

    return results

def write_eval_results(results, outpath):

    idx = set([model for model, _ in results.keys()])
    df = pd.DataFrame({key:[np.nan for i in idx] for key in set([metric for _, metric in results.keys()])}, index=idx)
    for model, metric in results.keys():
        df.loc[model, metric] = results[(model, metric)]
    df.to_csv(outpath)

def evaluate_2(targets, predictions):
    '''
    evaluate single target predictions
    predictions is a multiindex dict with 3 index levels: (model, target, dataset)
    '''
    # predictions = predictionsDict2df_2(predictions)
    results = {}

    for model in predictions.keys():
        
        for target in predictions[model].keys():
            
            y_true = targets[target]
            mask = y_true.notna()
            y_true = y_true[mask]
            
            for data_key in predictions[model][target].keys():
                y_score = pd.Series(predictions[model][target][data_key])
                
                mask = y_score.isnull().values
                if (mask.sum() > 0):
                    print('WARNING: prediction scores for model {0} contain {1} null values. Will be removed'.format(model, mask.sum()))
                    # y_score.fillna(replace_null_probs_with, inplace=True)
                y_score = y_score.loc[~mask]
                y_true = y_true.loc[~mask]

                auroc = roc_auc_score(y_true, y_score, average=None)
                results[(model, target, data_key, 'auroc')] = auroc
                
                auprc = average_precision_score(y_true, y_score, average=None)
                results[(model, target, data_key, 'auprc')] = auprc
    return results

def write_eval_results_2(results, outpath):
    
    '''
    results is a dict with keys: (model, target, dataset, metric)
    '''
    idx = set([(model, target, metric) for model, target, _, metric in results.keys()])
    df = pd.DataFrame({key:[np.nan for i in idx] for key in set([dataset for model, target, dataset, metric in results.keys()])}, index=idx)
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = ['model', 'target', 'metric']
    for model, target, _, metric in results.keys():
        for _, _, dataset, _ in results.keys():
            df.loc[(model, target, metric)].loc[dataset] = results[(model, target, dataset, metric)]
    df.to_csv(outpath)


def make_names_R_compatible(names = []):
    r_base = importr('base', lib_loc=R_libpath)
    return list(r_base.make_names(names))