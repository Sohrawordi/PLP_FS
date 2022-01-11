# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:45:51 2019

@author: Shaikot
"""
#import pandas as pd
def k_spaced_composition(data):
    import pandas as pd
    cols=[]
    amino_acid="ACDEFGHIKLMNPQRSTWYVX"
    for k in range(5):
        for i in range(len(amino_acid)):
            a=amino_acid[i]
            for j in range(len(amino_acid)):
                b=amino_acid[j]
                tem=""
                for t in range(k):
                    tem=tem+"x"
                #print(a+tem+b)
                cols.append("K_space_"+a+tem+b)
    #print(cols)
    df=pd.DataFrame(columns=cols)
    
    for long in range(len(data['Sequence'])):
        seq=data['Sequence'][long]
        #print(seq)    
        a=0
        #seq="AAAC"
        #A list of 21 Amino Acid
        
        row={}
        #amino_acid="XKP"
        for k in range(5):
            count=0
            for i in range (len(amino_acid)):
                a=amino_acid[i]
                for j in range(len(amino_acid)):
                    b=amino_acid[j]
                    c=0
                    for sl in range(len(seq)-k-1):
                        if seq[sl]==a:
                            if seq[sl+k+1]==b:
                                c=c+1
                                count=count+1
                            #print(a+b)
                        #print(a)
                     #a=a+1
                    tem=""
                
                    for m in range(k):
                         tem=tem+"x"
                    
                    #print("K_space_"+a+tem+b+"    "+str(c))
                    row["K_space_"+a+tem+b]=c
                    
            #print(count)
            #print(sum(row.values()))
            
            #the sum of all pairs
            totall=sum(row.values())
            
            for key in row:
                #print(key)
                row[key]=row[key]/totall
        #print(row)
             
            
        df=df.append(row,ignore_index=True)       
    return df       
            

           
          
import pandas as pd         
positive=pd.read_csv('Raw_Samples/Positive.csv')             
p=k_spaced_composition(positive)           
print(p)            
          
        
negative=pd.read_csv('Raw_Samples/Negative.csv')        
n=k_spaced_composition(negative)           
print(n)            
            
p.to_csv('Numerical_features/positive_numerical_k_space_samples.csv',index=False)

n.to_csv('Numerical_features/negative_numerical_k_space_samples.csv',index=False)            
            
            
            
            
            
            
            
            