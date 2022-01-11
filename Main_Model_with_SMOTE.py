# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:15:16 2021

@author: Md. Sohrawordi
"""


import pandas as pd
from collections import Counter
from sklearn.svm import SVC
data=pd.read_csv('Numerical_features/Balanced_data_class_label.csv')


y=data.pop("Class")


counter = Counter(y)
print("Sample Distribution", counter)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=10)

#Distribution of y predicted
print('Distribution of Train Data : \n' + str(pd.Series(y_train).value_counts()))

print('Distribution of Test Data : \n' + str(pd.Series(y_test).value_counts()))






"""
#parameter for C and gamma of SVM classifier
g=[]
for i in range(-8,9):
    g.append(2**i)

from sklearn.model_selection import GridSearchCV

sclf = SVC()
param_grid = [{'C':g, 'gamma':g, 'kernel': ['rbf']},]
clf = GridSearchCV(estimator=sclf, param_grid=param_grid, n_jobs=-1,cv=10,scoring = 'accuracy')
clf.fit(data,y)
print('Best score for data:  ', clf.best_score_) 

print('Best C:',clf.best_estimator_.C) 
print('Best Gamma:',clf.best_estimator_.gamma)









"""

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import matthews_corrcoef


"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(C=16, class_weight={0:1,1:1}, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=10,
                   n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.001, verbose=0,
                   warm_start=False)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =1,max_depth=2,random_state = 25)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 35)


from sklearn.tree import DecisionTreeClassifier

classifier= DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=1200)
"""







#classifier =SVC(kernel='rbf', C=64,gamma=0.25,probability=True, class_weight={0:1, 1:3})
classifier =SVC(kernel='rbf', C=64,gamma=0.25,probability=True, class_weight={0:1, 1:3})
y_class=y_train
x_data=X_train
cv = KFold(n_splits=10)

y_pred = cross_val_predict(classifier, x_data, y_class, cv = cv)

print(classification_report(y_class, y_pred))

print("Accuracy Score: "+str(accuracy_score(y_class,y_pred)))
print("Confusion_matrix :")


cm=confusion_matrix(y_class,y_pred)
print(cm)



TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
acc=(TP+TN)/(TP+TN+FP+FN)
sn=TP / (TP + FN)
sp =TN / (TN + FP)


print('Sensitivity : ', sn)


print('Specificity : ', sp)

print("Matthews correlation coefficient (MCC): ",matthews_corrcoef(y_class, y_pred))



#Distribution of y 
print('y actual : \n' +  str(pd.Series(y_class).value_counts()))
#Distribution of y predicted
print('y predicted : \n' + str(pd.Series(y_pred).value_counts()))


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_class, y_pred)

import numpy as np
#np.savetxt("ROC_Data/tpr_PLP_FS_train.txt",tpr,delimiter=',');
#np.savetxt("ROC_Data/fpr_PLP_FS_trian.txt",fpr,delimiter=',');


# calculate AUC
auc = roc_auc_score(y_class, y_pred)
print('AUC: %.3f' % auc)



#plot_roc_curve(fpr, tpr)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.05])
plt.ylim([0.0, 1.10])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 














classifier.fit(X_train, y_train) 


pred=classifier.predict(X_test) 
    
  
print(classification_report(y_test, pred))


print("Accuracy Score: "+str(accuracy_score(y_test,pred)))
print("Confusion_matrix :")


cm=confusion_matrix(y_test,pred)
print(cm)


TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
acc=(TP+TN)/(TP+TN+FP+FN)
sn=TP / (TP + FN)
sp =TN / (TN + FP)


print('Sensitivity : ', sn)


print('Specificity : ', sp)

print("Matthews correlation coefficient (MCC): ",matthews_corrcoef(y_test, pred))



# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, pred)


#np.savetxt("ROC_Data/tpr_PLP_FS_independent.txt",tpr,delimiter=',');
#np.savetxt("ROC_Data/fpr_PLP_FS_independent.txt",fpr,delimiter=',');
#np.savetxt("ROC_Data/thresholds_SMPP.txt",thresholds,delimiter=',');



# calculate AUC
auc = roc_auc_score(y_test, pred)
print('AUC: %.3f' % auc)

#plot_roc_curve(fpr, tpr)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.05])
plt.ylim([0.0, 1.10])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 















