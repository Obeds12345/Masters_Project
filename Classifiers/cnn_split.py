import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('helpers'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model, Sequential

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

from helpers.ProcessData import ProcessData
from helpers.scores import Scores
from helpers.average import Average
from Classifiers.classifiers import Classifiers

class CNN_SPLIT:
        
    def CNN(X, y, columns, threshold, counts, hidden_layers):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        scaler = StandardScaler().fit(X_test)
        X_test = scaler.fit_transform(X_test)
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
        in_shape = X_train.shape[1:]
        
        model = Classifiers().cnnModel_one_hidden(X_train, y_train) if hidden_layers == 1 else Classifiers().cnnModel_two_hidden(X_train, y_train)
        
        yhat_probs = model.predict(X_test, verbose=0)
        probas = np.array(yhat_probs)
        predictions = (probas > threshold).astype(np.int)
        
        p_main_score_avarage = precision_score(y_test, predictions, average='macro', zero_division=0)
        r_main_score_avarage = recall_score(y_test, predictions, average='macro', zero_division=0)
        f_main_score_avarage = f1_score(y_test, predictions, average='macro', zero_division=0)
        p_none_score_avarage = precision_score(y_test, predictions, average=None, zero_division=0)
        r_none_score_avarage = recall_score(y_test, predictions, average=None, zero_division=0)
        f_none_score_avarage = f1_score(y_test, predictions, average=None, zero_division=0)
        cm = Scores.get_tp_tn_fn_fp(y_test, predictions)
        test_count = len(y_test)
        
        my_columns = np.array(columns)
        results = [['main', p_main_score_avarage,  r_main_score_avarage, f_main_score_avarage, 0, 0, 0, 0, 0, 0, 0]]
        for L, P, R, F, C, CM in zip(my_columns, p_none_score_avarage, r_none_score_avarage, f_none_score_avarage, counts, cm):
            results.append([L, P, R, F, C, CM[0] + CM[2], CM[0], CM[1], CM[2], CM[3], test_count])
            
                
        df = pd.DataFrame(results, columns=['Label', 'Precision', 'Recall', 'F-Measure', 'lable in Data', 'lable in Test Data', 'TP', 'FP', 'FN', 'TN', 'Test Count' ])
        df = df.sort_values(by=['Precision'], ascending=False)
        df = df.reset_index(drop=True)    
        
        path = "./results/cnn/split/CNN_split_{0}_{1}_H_L.csv".format(threshold, hidden_layers)
        
        df.to_csv(path)
        print("*********************..........DONE SPLIT {0} {1} H_L ..........*********************".format(threshold, hidden_layers))
