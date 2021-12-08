import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('helpers'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

from helpers.ProcessData import ProcessData
from helpers.scores import Scores
from helpers.average import Average
from Classifiers.classifiers import Classifiers

class CUSTOM_FOLD_AVERAGE:
        
    def Custom(classifier, key_name, X, y, columns, n_of_fold, counts):
        kf = KFold(n_of_fold, True, 1)
        splits =  kf.get_n_splits(X)
    
        scores_all = {
            'custom' : {
                'main': { 'precision': [], 'recall':[], 'f-measure': []},
                'precision': [],
                'recall': [],
                'f-measure': [],
                'cm':[],
                'test_count':[]
                },
        }
            
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # print('len(y_test)', len(y_test))
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.fit_transform(X_train)
            scaler = StandardScaler().fit(X_test)
            X_test = scaler.fit_transform(X_test)

            # clf =  Classifiers().classifier(X_train, y_train)
            # importances = clf.coef_[0]
            # print(importances)
            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            
            p_main_score_avarage = precision_score(y_test, predictions, average='macro', zero_division=0)
            r_main_score_avarage = recall_score(y_test, predictions, average='macro', zero_division=0)
            f_main_score_avarage = f1_score(y_test, predictions, average='macro', zero_division=0)
            p_none_score_avarage = precision_score(y_test, predictions, average=None, zero_division=0)
            r_none_score_avarage = recall_score(y_test, predictions, average=None, zero_division=0)
            f_none_score_avarage = f1_score(y_test, predictions, average=None, zero_division=0)
            if (type(predictions).__module__ != np.__name__):
                predictions = predictions.toarray()
            cm = Scores.get_tp_tn_fn_fp(y_test, predictions)
        
            
            scores_all['custom']['main']['precision'].append(p_main_score_avarage)
            scores_all['custom']['main']['recall'].append(r_main_score_avarage)
            scores_all['custom']['main']['f-measure'].append(f_main_score_avarage)
            scores_all['custom']['precision'].append(p_none_score_avarage)
            scores_all['custom']['recall'].append(r_none_score_avarage)
            scores_all['custom']['f-measure'].append(f_none_score_avarage)
            scores_all['custom']['cm'].append(cm)
            scores_all['custom']['test_count'].append(len(y_test))
        
        results = {}
        for key in scores_all:
            p_main_score_avarage = Average.Average(scores_all[key]['main']['precision'])
            r_main_score_avarage = Average.Average(scores_all[key]['main']['recall'])
            f_main_score_avarage = Average.Average(scores_all[key]['main']['f-measure'])
            
            p_none_score_avarage = Average.NoneAverage(scores_all[key]['precision'])
            r_none_score_avarage = Average.NoneAverage(scores_all[key]['recall'])
            f_none_score_avarage = Average.NoneAverage(scores_all[key]['f-measure'])
            cm = Average.cmAverage(scores_all[key]['cm'])
            test_count = Average.Average(scores_all[key]['test_count'])
            
            my_columns = np.array(columns)
            results = [['main', p_main_score_avarage,  r_main_score_avarage, f_main_score_avarage, 0, 0, 0, 0, 0, 0, 0]]
            for L, P, R, F, C, CM in zip(my_columns, p_none_score_avarage, r_none_score_avarage, f_none_score_avarage, counts, cm):
                results.append([L, P, R, F, C, CM[0] + CM[2], CM[0], CM[1], CM[2], CM[3], test_count])


            df = pd.DataFrame(results, columns=['Label', 'Precision', 'Recall', 'F-Measure', 'lable in Data', 'lable in Test Data', 'TP', 'FP', 'FN', 'TN', 'Test Count' ])
            df = df.sort_values(by=['Precision'], ascending=False)
            df = df.reset_index(drop=True)    
            
            path = "./results/custom/fold_average/{}.csv".format(key_name)
            df.to_csv(path)

            print()
            print("*********************..........DONE FOLD AVERAGE {} ..........*********************".format(key_name))
            print()