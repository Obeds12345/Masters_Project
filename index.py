import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from Classifiers.cnn_fold_average import CNN_FOLD_AVERAGE
from Classifiers.cnn_fold import CNN_FOLD
from Classifiers.cnn_split import CNN_SPLIT

from Classifiers.custom_fold_average import CUSTOM_FOLD_AVERAGE
from Classifiers.custom_fold import CUSTOM_FOLD
from Classifiers.custom_split import CUSTOM_SPLIT

from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from helpers.ProcessData import ProcessData
from helpers.scores import Scores
 
if __name__=='__main__': 

    n_of_fold = 2
    X, y, columns = ProcessData.PrepareData()
    counts = [] 
    for i in range(y.shape[1]):
        y1 = Scores.column(y, i)
        counts.append(y1.count(1))
        
    classifiers = [
        # Random Classifier
        {  
            'classifier': BinaryRelevance(GaussianNB()), 
            'key': 'BinaryRelevance(GaussianNB())' 
        },
        # {   
        #     'classifier': BinaryRelevance(KNeighborsClassifier()), 
        #     'key': 'BinaryRelevance(KNeighborsClassifier())' 
        # },   
        # {  
        #     'classifier': BinaryRelevance(RandomForestClassifier(random_state=42)),
        #     'key': 'BinaryRelevance(RandomForestClassifier())' 
        # },    
        # {  
        #     'classifier': ClassifierChain(GaussianNB()), 
        #     'key': 'ClassifierChain(GaussianNB())' 
        # },
        # {   
        #     'classifier': ClassifierChain(KNeighborsClassifier()), 
        #     'key': 'ClassifierChain(KNeighborsClassifier())' 
        # },       
        # {  
        #     'classifier': ClassifierChain(RandomForestClassifier(random_state=42, n_estimators=100)),
        #     'key': 'ClassifierChain(RandomForestClassifier())' 
        # },
        # {  
        #     'classifier': OneVsRestClassifier(GaussianNB()), 
        #     'key': 'OneVsRestClassifier(GaussianNB())' 
        # },
        # {  
        #     'classifier': OneVsRestClassifier(SVC()),
        #     'key': 'OneVsRestClassifier(SVC())' 
        # },
        # {   
        #     'classifier': OneVsRestClassifier(KNeighborsClassifier()), 
        #     'key': 'OneVsRestClassifier(KNeighborsClassifier())' 
        # },
        # {  
        #     'classifier': OneVsRestClassifier(RandomForestClassifier(random_state=42, n_estimators=100)),
        #     'key': 'OneVsRestClassifier(RandomForestClassifier())' 
        # }, 
  
    ]
            
         
    # for classifier in classifiers:
        # CUSTOM_SPLIT.Custom(classifier.get('classifier'), classifier.get('key'), X, y, columns, counts)
        # CUSTOM_FOLD_AVERAGE.Custom(classifier.get('classifier'), classifier.get('key'), X, y, columns, 10, counts)
        # CUSTOM_FOLD.Custom(classifier.get('classifier'), classifier.get('key'), X, y, columns, 10, counts)
    
    CNN_SPLIT.CNN(X, y, columns, 0.3, counts, 1)
    # CNN_FOLD_AVERAGE.CNN(X, y, columns, 0.3, 10, counts, 1)
    # CNN_FOLD.CNN(X, y, columns, 0.3, 10, counts, 1)
    
    # record time
    
    
# {  
#     'classifier': MultiOutputClassifier(GaussianNB()), 
#     'key': 'MultiOutputClassifier(GaussianNB())' 
# },
# {   
#     'classifier': MultiOutputClassifier(KNeighborsClassifier()), 
#     'key': 'MultiOutputClassifier(KNeighborsClassifier())' 
# },
# {  
#     'classifier': MultiOutputClassifier(RandomForestClassifier(random_state=42, n_estimators=100)),
#     'key': 'MultiOutputClassifier(RandomForestClassifier())' 
# },