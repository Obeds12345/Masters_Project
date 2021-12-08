import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('helpers'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

from helpers.ProcessData import ProcessData
from helpers.scores import Scores
from helpers.average import Average
from Classifiers.classifiers import Classifiers

class CUSTOM_FOLD:
    
    def Custom(classifier, key_name, X, y, columns, n_of_fold, counts):
        Labels = [
            'LI1',
            'CM',
            'I1',
            'F1',
            'SP1',
            'L2',
            'FP5',
            'OTHER',
            'LI2',
            'I7',
            'F2',
            'MM7',
            'CP5',
            'FP1',
            'MM6',
            'I6',
            'CP4',
            'F3',
            'MM1',
            'I8',
            'SP4',
            'T3',
            'main',
            'L1',
            'NB1',
            'MM5',
            'I5',
            'W1',
            'CP1',
            'FP2',
            'MM9',
            'CC4',
            'FP6',
            'PB1',
            'CP2',
            'I3',
            'AP3',
            'L12',
            'O1',
            'PN6',
            'NB4',
            'NB5',
            'PB6',
            'R4',
            'PN1',
            'SA1',
            'RD1',
            'RD2',
            'RD7',
            'S2',
            'SP2',
            'SP3',
            'SS1',
            'SS2',
            'SS4',
            'SS6',
            'SS8',
            'SS9',
            'T1',
            'T2',
            'W2',
            'NB3',
            'LI1F2',
            'N1',
            'FP3',
            'AP2',
            'AR9',
            'C1',
            'CC1',
            'CC2',
            'CC3',
            'CM1',
            'D2',
            'D3',
            'E1',
            'F4',
            'F6',
            'FC1',
            'FP4',
            'MM8',
            'H1',
            'H2',
            'H3',
            'I2',
            'L3',
            'L7',
            'AP1',
            'LITERACY',
            'LL1',
            'LL2',
            'MM2',
            'MM3',
            'MM4',
            'W3'
        ]
        
        kf = KFold(n_of_fold, True, 1)
        splits =  kf.get_n_splits(X)
    
        scores_all = {
            'custom' : {
                'main': { 'precision': [], 'recall':[] },
                'precision': [],
                'recall': [],
                'cm':[],
                'test_count':[]
                },
        }

        my_columns = np.array(columns)
        results = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
 
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.fit_transform(X_train)
            scaler = StandardScaler().fit(X_test)
            X_test = scaler.fit_transform(X_test)

     
            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            # cr_y1 = classification_report(y_test, predictions)
            # print(cr_y1)
            
            # main_p_score = precision_score(y_test, predictions, average='macro', zero_division=0)
            # main_r_score = recall_score(y_test, predictions, average='macro', zero_division=0)
            # p_score = precision_score(y_test, predictions, average=None, zero_division=0)
            # r_score = recall_score(y_test, predictions, average=None, zero_division=0)
            
            p_main_score_avarage = precision_score(y_test, predictions, average='macro', zero_division=0)
            r_main_score_avarage = recall_score(y_test, predictions, average='macro', zero_division=0)
            f_main_score_avarage = f1_score(y_test, predictions, average='macro', zero_division=0)
           
            p_none_score_avarage = precision_score(y_test, predictions, average=None, zero_division=0)
            r_none_score_avarage = recall_score(y_test, predictions, average=None, zero_division=0)
            f_none_score_avarage = f1_score(y_test, predictions, average=None, zero_division=0)
           
            
            if (type(predictions).__module__ != np.__name__):
                predictions = predictions.toarray()
                
            cm = Scores.get_tp_tn_fn_fp(y_test, predictions)
            for L, P, R, F, C, CM in zip(my_columns, p_none_score_avarage, r_none_score_avarage, f_none_score_avarage, counts, cm):
                results.append([L, P, R, F, C, CM[0] + CM[2], CM[0], CM[1], CM[2], CM[3], len(y_test)])

        for key in scores_all:
            results_main = []
            
            for row in Labels:
                a = np.array(results)
                values = np.array([row])
                np_indecies =  np.where(np.isin(a[:,0], values))
                for indecies in np_indecies:
                    for index in indecies:
                        results_main.append(results[index])

            df = pd.DataFrame(results_main, columns=['Label', 'Precision', 'Recall', 'F-Measure', 'lable in Data', 'lable in Test Data', 'TP', 'FP', 'FN', 'TN', 'Test Count' ])
                         
            # df = df.sort_values(by=['Precision'], ascending=False)
            # df = df.reset_index(drop=True)    
            path = "./results/custom/fold/{}.csv".format(key_name)
            df.to_csv(path)
            
            print()
            print("*********************..........DONE FOLD {} ..........*********************".format(key_name))
            print()
        