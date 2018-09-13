# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:02:35 2018

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

df = pd.read_csv('targets_w_physicians2.csv',index_col='case')

## replace all positions contain "fellow" with "fellow"
df.position.mask(df.position.str.contains('fellow'), 'fellow', inplace=True)

## chi2 when instances with null values are removed
chi2_pos = df.drop(df.columns[[0,1]],axis=1).apply(lambda x: chi2(df.position, x))

chi2_pos.describe()

## calculate # labels for each target
ones = df.drop(['physician', 'position'], axis=1).sum(axis=0)
ones.describe()
ones[ones > 0].describe()

## number of available (non-nulls) for each target
nns = df.drop('physician', axis=1).count(axis=0)
nns.describe()

## proportion of labeled within availbles for each target
ones_nns = ones/nns
ones_nns[ones_nns > 0].describe()
ones_nns[ones_nns >= .1].describe()
