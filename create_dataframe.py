# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:11:34 2018

@author: mot16
"""
import numpy as np
import pandas as pd
#from scipy import stats

## write all_feature_names as text
fnames = np.load("all_feature_names.npy").astype(str)
np.savetxt('all_feature_names.txt',fnames, '%s')


## write physician_patient map into a file with only 178 patients from training data
## the physician_patient map has more patient ids than training data
ph_pt_map = np.loadtxt('labeled_case_list.txt', 'U', ' ')
pt = np.loadtxt('case_order_rows.txt', 'U')
valid_map = ph_pt_map[np.isin(ph_pt_map[:,1], pt)]
np.savetxt('labeled_case_list_178.txt', valid_map, '%s')

## read target_present_case_rows.txt into a dict
with open('target_present_case_rows.txt') as f:
    lines = f.readlines()

d = {}  
for i in range(len(lines)):
    l = lines[i].split('\n')[0]
    d[i+1] = [] if len(l) == 0 else [int(x) for x in l.split(',')]

## read target matrix and set missing ones as nan, instead of 0    
target = np.loadtxt('targets.txt', 'float16', delimiter=',')
for t_idx in d.keys():
    idx_to_none = np.isin(np.array(range(target.shape[0])), np.array(d[t_idx])-1, invert=True)
    target[idx_to_none, t_idx-1] = np.nan
    
np.savetxt('targets_w_nan.txt', target, '%1.f', delimiter=',')


## create dataframe from cases, physicians, and targets
with open('target_names.txt') as f:
    target_names = f.readlines()
target_names = [x.split('\n')[0] for x in target_names]
df = pd.DataFrame(target, columns=target_names, index=pt)

m = pd.DataFrame(valid_map)
m.index = m.iloc[:,1]
m.drop(1,axis=1,inplace=True)
m.columns=['physician']
df = df.join(m)

p = pd.read_csv('physicians.txt')
p.position = p.position.str.lower().str.strip()
p.rename(columns={'years since med school':'years_since_med_school','years of ICU experience':'years_ICU_experience'}, inplace=True)
p.years_since_med_school = p.years_since_med_school.str.strip().str.replace('yrs', '')
p.years_ICU_experience = p.years_ICU_experience.str.strip().str.replace(r'[yrs|yr]', '').str.replace('<1', '0')
p.years_since_med_school[p.ID == 'M2'] = 8
p.position[p.ID == 'S23'] = 'fellow 2nd'

df['position'] = df.physician.apply(lambda x: p.position[p.ID == x].item())
df['years_since_med_school'] = df.physician.apply(lambda x: p.years_since_med_school[p.ID == x].item())
df['years_ICU_experience'] = df.physician.apply(lambda x: p.years_ICU_experience[p.ID == x].item())

cols = df.columns.tolist()
cols = cols[-4:] + cols[:-4]
df = df[cols]

df.to_csv('targets_w_physicians.csv')
