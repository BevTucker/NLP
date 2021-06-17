# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:53:37 2020

@author: Michael Cai

"""
# =============================================================================
# PACKAGES LOADING
# =============================================================================

import pandas as pd
import numpy as np
import sys

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics 
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix as plcm
import seaborn as sns

# =============================================================================
# MODELING DATASETS CREATION
# =============================================================================

# create a funciton to evaluate prediction accuracy w/ different datasets
# =============================================================================
def Eval(df,col,test_size):
    df_model = df.copy()
    df_model = shuffle(df_model)
    train_x, test_x, train_y, test_y = train_test_split(df_model.iloc[:,1:],df_model[col],test_size=test_size,random_state=1)
    clf.fit(train_x, train_y)
    predict = clf.predict(test_x)
   
    # compute log ratios of log_pos_ratio - log_neg_ratio for most indicative features
    df_if = pd.DataFrame(list(clf.feature_log_prob_[1] - clf.feature_log_prob_[0]), list(df_model.columns[1:]), columns=['log_ratios'])
    
    # plot a confusion matrix of the prediction results
    disp = plcm(clf, test_x, test_y
            , display_labels = None
            , cmap = plt.cm.Blues
            , normalize = None)
    disp.ax_.set_title('Matrix of Prediction Results')
    print('\nMatrix of Prediction Results')
    print(disp.confusion_matrix)
    plt.show()
    
    print("\nThe prediction accuracy on '{}' is {}%.".format(col, round(metrics.accuracy_score(test_y, predict)*100, 3))) # print accuracy result
    # print('\nTop 20 informative words in Category A:\n', df_if.sort_values(by='log_ratios', ascending=False).head(20)) # print top 20 indicative words in Cat-A
    # print('\nTop 20 informative words in Category B:\n', df_if.sort_values(by='log_ratios', ascending=True).head(20)) # print top 20 indicative words in Cat-B
    
# End of function.
# =============================================================================


# create a funciton to evaluate prediction accuracy in k-fold w/ different datasets
# =============================================================================
def kEval(df, col, k):
    df_model = df.copy()
    df_model = shuffle(df_model)
    x = df_model.iloc[:,1:]
    y = df_model[col]
    a = list(cross_val_score(clf, x, y, cv=k, n_jobs=-1))
    print("\nThe {}-fold cross-validation results are: \n\t{}" .format(k,a))
    print("\nThe average prediction accuracy on '{}', from a {}-fold cross-validation, is {}%.".format(col, k, round((sum(a)/len(a) * 100),3)))
    
# End of function.
# =============================================================================





# =============================================================================
# PREDICTION & MODELING
# =============================================================================

# 1) Bernoulli Naive Bayes classifier w/ binary data
# =============================================================================

# define classifier and parameters
clf = BernoulliNB(alpha=1, binarize=0.0, class_prior=None, fit_prior=True)


# perform prediction and evaluate performance
Eval(dfv_bi_lem,'Labels',0.3)
Eval(dfv_bi_lem_vb_adj_adv,'Labels',0.3)

# evaluate performance with 5-fold cross-validation
kEval(dfv_bi_lem,'Labels',3)
kEval(dfv_bi_lem_vb_adj_adv,'Labels',3)


# 2) Multinomial Naive Bayes classifier w/ word frequencies, L1 norm, and L2 norm
# =============================================================================

# store 3 datasets in a list
datasets = [dfv_tf_lem, dfv_tf_lem_nn_adj, dfv_tf_lem_vb_adj_adv]

# define classifier and parameters
clf = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)

# perform prediction and evaluate performances on 3 datasets with a for-loop statements
for data in datasets:
    Eval(data,'Labels',0.3)


# evaluate performances with 5-fold cross-validation on 3 datasets with a for-loop statements
for data in datasets:
    kEval(data,'Labels',5)