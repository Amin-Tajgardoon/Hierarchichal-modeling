# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:40:33 2018

@author: mot16
"""
import pandas as pd

df = pd.read_csv("targets_w_physicians.csv")

## What proportion of available targets were labeled by each physician?
sums = df.groupby('physician', sort=False).apply(sum).drop(df.columns[0:4], 1)
counts = df.groupby('physician', sort=False).apply(lambda x: x.count()).drop(df.columns[0:4], 1)

c = counts.sum(axis=1)
s = sums.sum(axis=1)
p = s/c
ph_stats = pd.DataFrame({"#available_targets":c,"#labeled_targets":s,
                         "proportion_labeld":s/c,
                         "#cases_labeled":df.physician.value_counts(sort=False)},
                         columns=["#available_targets","#labeled_targets",
                         "proportion_labeld", "#cases_labeled"],
                         index=c.index)
ph_stats.to_csv('physician_stats.csv')

plt = p.sort_values(ascending=False).plot(kind='bar')
plt.set_ylabel('labeling propotion')
plt.set_title('labeling proportion for each physician')

## available targets and labels for each case

row_sums = df.iloc[:,4:].sum(axis=1)
case_label_stat = row_sums.describe()
case_label_stat['count_zero'] = sum(row_sums == 0)
case_label_stat['count_null'] = sum((row_sums.isnull()))

row_counts = df.iloc[:,4:].count(axis=1)
case_target_stat = row_counts.describe()
case_target_stat['count_zero'] = sum(row_counts == 0)

case_label_target_prop = row_sums / row_counts
t = pd.DataFrame([case_label_stat, case_target_stat, case_label_target_prop.describe()]).T
t.rename(columns={0:'label', 1:'available_target',2:'label_target_prop'}, inplace=True)
t.to_csv('case_label_target_stats.csv')

no_label_cases = df[(row_sums.isnull()) | (row_sums == 0)]
nl = no_label_cases.iloc[:,4:].count(axis=1)
nl.name = 'available_target_count'
pd.DataFrame([no_label_cases.physician, nl]).T.to_csv('cases_with_no_labels.csv')