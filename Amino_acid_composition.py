# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:28:33 2020

@author: Shaikot
"""


def amino_acid_composition(dataset):
    import pandas as pd
    #A list of 21 Amino Acid
    amino_acid="ACDEFGHIKLMNPQRSTWYVX"
    colls=[]  
    #sample = pd.read_csv(dataset)
    
    for i in range (len(amino_acid)):
        colls.append("Amino_acid_"+amino_acid[i])
    #print(colls)
    df=pd.DataFrame(columns=colls)
    for i in range(len(dataset['Sequence'])):
        #print(i)
        row={}
        for acid in amino_acid:
            #print(seq.count(acid))
            row["Amino_acid_"+acid]=dataset['Sequence'][i].count(acid)/len(dataset['Sequence'])
        df=df.append(row,ignore_index=True)

    



    return df







import pandas as pd
positive=pd.read_csv('Raw_Samples/Positive.csv')

pdf=amino_acid_composition(positive)

negative=pd.read_csv('Raw_Samples/Negative.csv')
#print(ndf)
ndf=amino_acid_composition(negative)


pdf.to_csv('Numerical_features/positive_numerical_aac_samples.csv',index=False)

ndf.to_csv('Numerical_features/negative_numerical_aac_samples.csv',index=False)





















