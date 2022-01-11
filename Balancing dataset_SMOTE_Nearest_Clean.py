# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:17:40 2021

@author: Md. Sohrawordi
"""



import pandas as pd
from collections import Counter

aac_positive=pd.read_csv('Numerical_features/positive_numerical_aac_samples.csv')
aac_negative=pd.read_csv('Numerical_features/negative_numerical_aac_samples.csv')

binary_positive=pd.read_csv('Numerical_features/positive_numerical_binary_samples.csv')
binary_negative=pd.read_csv('Numerical_features/negative_numerical_binary_samples.csv')

kspace_positive=pd.read_csv('Numerical_features/positive_numerical_k_space_samples.csv')
kspace_negative=pd.read_csv('Numerical_features/negative_numerical_k_space_samples.csv')


psample=binary_positive.copy()
for col in aac_positive.columns: 
    psample[col]=aac_positive[col] 

for col in kspace_positive.columns: 
    psample[col]=kspace_positive[col]



nsample=binary_negative.copy()
for col in aac_negative.columns: 
    nsample[col]=aac_negative[col] 

for col in kspace_negative.columns: 
    nsample[col]=kspace_negative[col]



class_label=[]
for i in range (len(psample.index)):
    class_label.append(1)





for i in range (len(nsample.index)):
    class_label.append(0)

counter = Counter(class_label)
print("Orginal Samples",counter)

sample=pd.concat([psample,nsample], ignore_index=True, sort =False)






from imblearn.under_sampling import NeighbourhoodCleaningRule
undersample = NeighbourhoodCleaningRule(n_neighbors=21, threshold_cleaning=0.2)






# transform the dataset

X, y = undersample.fit_resample(sample,class_label)
# summarize the new class distribution
counter = Counter(y)
print("After Downsampling", counter)


from imblearn.over_sampling import SMOTE
x_r, y_r = SMOTE(sampling_strategy=1/2).fit_resample(X, y)


counter = Counter(y_r)
print("After Over Sampling Samples",counter)


aac_cols=aac_positive.columns
aac_balance=x_r[aac_cols]

binary_cols=binary_positive.columns

binary_balance=x_r[binary_cols]



kspace_cols=kspace_positive.columns

kspace_balance=x_r[kspace_cols]






#F_score feature selecton techniques for amino acid composition

x_data=aac_balance
y=y_r


from sklearn.feature_selection import SelectKBest, f_classif
# Create and fit selector
selector = SelectKBest(f_classif, k=3)  
selector.fit(x_data,y)
cols_index = selector.get_support(indices=True)


aac_df=pd.DataFrame()
aac_col=x_data.columns
for col in cols_index: 
    #print(aac_col[col])#   nsample[col]=kspace_negative[col]
    aac_df[aac_col[col]]=x_data[aac_col[col]]



#F_score feature selecton techniques for binary encoding

x_data=binary_balance
y=y_r

# Create and fit selector
selector = SelectKBest(f_classif, k=30)  
selector.fit(x_data,y)
cols_index = selector.get_support(indices=True)


binary_df=pd.DataFrame()
binary_col=x_data.columns
for col in cols_index: 
    #print(aac_col[col])#   nsample[col]=kspace_negative[col]
    binary_df[binary_col[col]]=x_data[binary_col[col]]





#F_score feature selecton techniques for kspace encoding

x_data=kspace_balance
y=y_r

# Create and fit selector
selector = SelectKBest(f_classif, k=31)  
selector.fit(x_data,y)
cols_index = selector.get_support(indices=True)


kspace_df=pd.DataFrame()
kspace_col=x_data.columns
for col in cols_index: 
    #print(aac_col[col])#   nsample[col]=kspace_negative[col]
    kspace_df[kspace_col[col]]=x_data[kspace_col[col]]




optimal_x=binary_df.copy()
for col in aac_df.columns: 
    optimal_x[col]=aac_df[col] 

for col in kspace_df.columns: 
    optimal_x[col]=kspace_df[col]


optimal_x["Class"]=y
optimal_x.to_csv('Numerical_features/Balanced_data_class_label.csv',index=False)



























