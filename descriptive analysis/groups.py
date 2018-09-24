# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:30:42 2018

@author: mot16
"""

## according to Andy, the targets that do not match to the rootgroups table,
## are medications. The one that match have 6 or less chatacters.
## However, I see target names like glucose which is suposed to be GLU in targets
##
 
import numpy as np
import pandas as pd

## read variable groups (domains) table
groups = pd.read_csv('rootgroup.csv')
df = pd.read_csv('targets_w_physicians.csv',index_col='case')

sum(df.columns.str.lower().isin(groups.root.str.lower().append(groups.labname.str.lower())))
df.columns[(df.columns.str.lower().isin(groups.labname.str.lower())) & ~(df.columns.str.lower().isin(groups.root.str.lower()))]

i_idx = df.columns.str.lower().isin(groups.root.str.lower().append(groups.labname.str.lower()))
grt6 = df.columns[(df.columns.to_series().str.strip().str.len() > 6) & (i_idx)]


groups.groupname.unique().shape
groups.groupname.value_counts()

groups.groupname[groups.root.str.lower().isin(df.columns.str.lower())].value_counts()
groups.groupname[groups.root.str.lower().isin(df.columns.str.lower())].value_counts().shape
g2 = groups.groupname[groups.root.str.lower().isin(df.columns.str.lower())].unique()
g1 = groups.groupname.unique()
g1[~g2.isin(g1)]
g1[~g1.isin(g2)]
g1[~np.isin(g1,g2)]