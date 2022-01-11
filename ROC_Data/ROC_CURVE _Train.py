# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 01:12:07 2020

@author: Shaikot
"""


#from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import numpy as np

#print(proba)

tpr_PLP_FS= np.loadtxt("tpr_PLP_FS_train.txt",delimiter=",")
fpr_PLP_FS= np.loadtxt("fpr_PLP_FS_train.txt",delimiter=",")
plt.plot(fpr_PLP_FS,tpr_PLP_FS,label="PLP_FS, AUC="+str(99.96))

tpr_RAM_PGK= np.loadtxt("RAM-PGK_tpr_train.txt",delimiter=",")
fpr_RAM_PGK= np.loadtxt("RAM-PGK_fpr_train.txt",delimiter=",")
plt.plot(fpr_RAM_PGK,tpr_RAM_PGK,label="RAM-PGK, AUC="+str(85.12))


tpr_iDPGK= np.loadtxt("iDPGK_tpr_train.txt",delimiter=",")
fpr_iDPGK= np.loadtxt("iDPGK_fpr_trian.txt",delimiter=",")
plt.plot(fpr_iDPGK,tpr_iDPGK,label="iDPGK, AUC="+str(81.18))

tpr_Bigram_PGK= np.loadtxt("Bigram_PGK_tpr_train.txt",delimiter=",")
fpr_Bigram_PGK= np.loadtxt("Bigram_PGK_fpr_train.txt",delimiter=",")
plt.plot(fpr_Bigram_PGK,tpr_Bigram_PGK,label="Bigram_PGK, AUC="+str(82.95))

tpr_PhoglyPred= np.loadtxt("PhoglyPred_tpr_train.txt",delimiter=",")
fpr_PhoglyPred= np.loadtxt("PhoglyPred_fpr_train.txt",delimiter=",")
plt.plot(fpr_PhoglyPred,tpr_PhoglyPred,label="PhoglyPred, AUC="+str(995.59))

tpr_Phogly_PseAAC= np.loadtxt("Phogly_PseAAC_tpr_train.txt",delimiter=",")
fpr_Phogly_PseAAC= np.loadtxt("Phogly_PseAAC_fpr_train.txt",delimiter=",")
#thresholds_CKSAAP_FormSite=np.loadtxt("thresholds_CKSAAP_FormSite.txt",delimiter=",")
plt.plot(fpr_Phogly_PseAAC,tpr_Phogly_PseAAC,label="Phogly_PseAAC, AUC="+str(89.36))




plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.05])
plt.ylim([0.0, 1.10])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")






"""
probs = proba[:, 1]

fpr, tpr, thresholds = roc_curve(y_class, probs)

#plot_roc_curve(fpr, tpr)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.05])
plt.ylim([0.0, 1.10])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend()

"""