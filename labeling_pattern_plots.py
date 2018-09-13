# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:18:12 2018

@author: mot16
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def chi2(x1, x2):
    if x1.count() * x2.count() == 0:
        return np.nan
    _, p, _,_ = chi2_contingency(pd.crosstab(x1, x2))
    return p

def get_label_proportions(df):    
    labeled_count_per_phy = df.drop('physician',axis=1).apply(lambda x : df.physician[x == 1].value_counts()).dropna(axis=1,how='all')    
    nns_grp_by_phs = df.groupby('physician', sort=False).count()
    targets_all_ph_labeled = labeled_count_per_phy.columns[labeled_count_per_phy.count(axis=0) == df.physician.unique().shape[0]]
    prop_labeled_per_phy = labeled_count_per_phy[targets_all_ph_labeled] / nns_grp_by_phs[targets_all_ph_labeled]
    return prop_labeled_per_phy

df = pd.read_csv('targets_w_physicians.csv',index_col='case')

### chi2 when instances with null values are removed
chi2_results = df.iloc[:,1:].apply(lambda x: chi2(df.physician, x))

#chi2_results.describe()
#ones[(chi2_results <= .1) & ones_nns > 0].sort_values()

## plot targets labeled by all physicians and the counts for each physician


prop_labeled_per_phy = get_label_proportions(df)

plt = prop_labeled_per_phy.plot(kind='bar', width=.8, rot=0, figsize=(25,10))
labels = prop_labeled_per_phy.columns.values.copy()
labels[chi2_results[labels] <= 0.1] += '*'
plt.legend(loc=(0,-0.2),title='target variables (*=significantly associated with physicians. chi2, p-value<=0.1)',labels=labels, mode="expand", ncol=9)
plt.set_xlabel('Physician', fontsize='20')
plt.set_ylabel('proportion of labeled', fontsize='20')
plt.set_title('Proportion of targets labeled by each physician (only for targets that were labeled by all physicians)', fontsize='20')

## physicians labeling patterns for each diagnosis

diag = pd.read_csv('diagnosis_features.txt', header=None)
diag = diag.iloc[:,0]
df2 = df.copy()
df2['diag'] = diag.values

diag_0 = df2.drop('diag', axis=1).loc[df2.diag == 0,:]
prop_labeled_per_phy0 = get_label_proportions(diag_0)
chi2_results_0 = diag_0.drop('physician',axis=1).apply(lambda x: chi2(diag_0.physician, x))

plt = prop_labeled_per_phy0.plot(kind='bar', width=.8, rot=0, figsize=(25,10))
labels = prop_labeled_per_phy0.columns.values.copy()
labels[chi2_results_0[labels] <= 0.1] += '*'
plt.legend(loc=(0,-0.2), ncol=6, title='target variables (*=significantly associated with physicians. chi2, p-value<=0.1)',labels=labels, mode="expand")
plt.set_xlabel('Physician', fontsize='20')
plt.set_ylabel('proportion of labeled', fontsize='20')
plt.set_title('Diagnosis 0 \n Proportion of targets labeled by each physician (only for targets that were labeled by all physicians)', fontsize='20')

labeled_count_per_phy = df.drop('physician',axis=1).apply(lambda x : df.physician[x == 1].value_counts()).dropna(axis=1,how='all')    
nns_grp_by_phs = df.groupby('physician', sort=False).count()
targets_all_ph_labeled = labeled_count_per_phy.columns[labeled_count_per_phy.count(axis=0) == df.physician.unique().shape[0]]
prop_labeled_per_phy = labeled_count_per_phy[targets_all_ph_labeled] / nns_grp_by_phs[targets_all_ph_labeled]


diag_1 = df2.drop('diag', axis=1).loc[df2.diag == 1,:]
prop_labeled_per_phy1 = get_label_proportions(diag_1)
chi2_results_1 = diag_1.drop('physician',axis=1).apply(lambda x: chi2(diag_1.physician, x))

plt = prop_labeled_per_phy1.plot(kind='bar', width=.8, rot=0, figsize=(25,10))
labels = prop_labeled_per_phy1.columns.values.copy()
labels[chi2_results_1[labels] <= 0.1] += '*'
plt.legend(loc=(0,-0.2), ncol=6, title='target variables (*=significantly associated with physicians. chi2, p-value<=0.1)',labels=labels, mode="expand")
plt.set_xlabel('Physician', fontsize='20')
plt.set_ylabel('proportion of labeled', fontsize='20')
plt.set_title('Diagnosis 1 \n Proportion of targets labeled by each physician (only for targets that were labeled by all physicians)', fontsize='20')

## labeling proportion for years_of_ICU_experience
df['years_ICU_experience_cat'] = df.years_ICU_experience
df.years_ICU_experience = df.years_ICU_experience.astype(np.int8)
df.years_ICU_experience_cat[df.years_ICU_experience > 2] = "gr_2"
df.years_ICU_experience_cat[df.years_ICU_experience <=2] = "leq_2"

label_count_per_experience = df.drop(df.columns[0:4], axis=1).apply(
        lambda col : df.years_ICU_experience_cat[col == 1].value_counts()).dropna(axis=1,how='all')
nns_grp_by_experience = df.drop(df.columns[0:3], axis=1).groupby(
        'years_ICU_experience_cat', sort=False).count()
targets_all_labeled = label_count_per_experience.columns[
        label_count_per_experience.count(
                axis=0) == df.years_ICU_experience_cat.unique().size]
prop_labeled_per_experience = label_count_per_experience[
        targets_all_labeled] / nns_grp_by_experience[targets_all_labeled]

chi2_results_exp = df.drop(df.columns[0:4],axis=1).apply(
        lambda x: chi2(df.years_ICU_experience_cat, x))

prop_labeled_per_experience = prop_labeled_per_experience.reindex(
        ['leq_2','gr_2'])
prop_labeled_per_experience.rename(
        index={'gr_2':'more_than_2','leq_2': 'less_or_equal_2'}, inplace=True)

plt = prop_labeled_per_experience.plot(
        kind='bar', width=.9, rot=0, figsize=(25,10))
labels = prop_labeled_per_experience.columns.values.copy()
labels[chi2_results_exp[labels] <= 0.1] += '*'
plt.legend(loc=(0,-.4), ncol=9,
           title='target variables (*=significantly associated with ' +
           'ICU experience level. chi2, p-value<=0.1)',labels=labels, mode="expand")
plt.set_xlabel('ICU experience level', fontsize='20')
plt.set_ylabel('proportion of labeled targets', fontsize='20')
plt.set_title('Proportion of targets labeled by physicians ' +
              'at each ICU experience level ' +
              '(only for targets that were labeled by all physicians)',
              fontsize='20')


plt = prop_labeled_per_experience.loc[:,chi2_results_exp[
        prop_labeled_per_experience.columns] <= .1].plot(
        kind='bar', width=.8, rot=0, figsize=(25,10))
plt.legend(loc=(0,-.2), ncol=9,
           title='target variables that are significantly associated with ' +
           'ICU experience level (chi2, p-value<=0.1)', mode="expand")
plt.set_xlabel('ICU experience level', fontsize='20')
plt.set_ylabel('proportion of labeled targets', fontsize='20')
plt.set_title('Proportion of targets labeled by physicians ' +
              'at each ICU experience level ' +
              '(only for significantly correlated targets that were labeled by all physicians)',
              fontsize='20')

