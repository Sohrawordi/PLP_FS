# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:56:02 2019

@author: Shaikot
"""




##positive_sample = pd.read_csv('Dataset/Positive_Samples.csv')
#negative_sample = pd.read_csv('Dataset/Negative_Samples.csv')


def binary_data(positive_sample):
    import pandas as pd
    amino_acid="ACDEFGHIKLMNPQRSTWYVX"
    binary_code={}
    colls=[]    
    for i in range (len(amino_acid)*len(positive_sample['Sequence'][0])):
        colls.append("Binary"+str(i))

    for i in amino_acid:
        a=[]
        for j in amino_acid:
            if i==j:
                a.append(1)
            else:
                a.append(0)  
        binary_code[i]=a 
        
        
        
    pdf=pd.DataFrame(columns=colls)
    for k in range(len(positive_sample['Sequence'])):
        c=0 
        row_value={}
        for i in positive_sample['Sequence'][k]:
            for j in binary_code[i]:
                row_value["Binary"+str(c)]=j
                c=c+1
        ##row_value["Class"]=1        
        pdf=pdf.append(row_value,ignore_index=True) 
    
    return pdf


import pandas as pd
positive=pd.read_csv('Raw_Samples/Positive.csv')

pdf=binary_data(positive)
negative=pd.read_csv('Raw_Samples/Negative.csv')
#print(ndf)
ndf=binary_data(negative)


pdf.to_csv('Numerical_features/positive_numerical_binary_samples.csv',index=False)

ndf.to_csv('Numerical_features/negative_numerical_binary_samples.csv',index=False)




     

    
    
    
    
    
    

      
        
        
        