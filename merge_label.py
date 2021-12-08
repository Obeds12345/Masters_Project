import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('helpers'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


if __name__=='__main__':
    df1 = pd.read_csv('./results/custom/fold_average/BinaryRelevance(GaussianNB()).csv')
    df2 = pd.read_csv('./results/custom/fold_average/ClassifierChain(GaussianNB()).csv')
    df3 = pd.read_csv('./results/custom/fold_average/OneVsRestClassifier(GaussianNB()).csv')
    # df4 = pd.read_csv('./results/custom/fold_average/CNN.csv')
    
    columns = ['Precision', 'Recall', 'F-Measure', 'TP', 'FP', 'FN', 'TN']
    classifiers = ["BR_NB", "CC_NB", "OVR_NB"]
    # classifiers = ["BinaryRelevance", "ClassifierChain"]
    
    df1 = df1[df1.Precision != 0]
    df2 = df2[df2.Precision != 0]
    df3 = df3[df3.Precision != 0]
    
    
    dataframes = [df1, df2, df3]
    
    for index, df in enumerate(dataframes):
        classifier = classifiers[index]
        # df = df[df["F-Measure"] != 0]
        # df = df[df.Recall != 0]
        # df = df[df.Precision != 0]
        df.rename(columns = {'Precision': '{}_Precision'.format(classifier),
                             'Recall': '{}_Recall'.format(classifier), 
                             'F-Measure': '{}_F-Measure'.format(classifier), 
                             'lable in Data': '{}_lable in Data'.format(classifier), 
                             'TP': '{}_TP'.format(classifier), 
                             'FP': '{}_FP'.format(classifier), 
                             'FP': '{}_FP'.format(classifier), 
                             'FN': '{}_FN'.format(classifier), 
                             'TN': '{}_TN'.format(classifier), 
                             'Test Count': '{}_Test Count'.format(classifier)}, 
                            inplace=True)
        

    # pd.merge(df1, df2, on="movie_title")
    result = dataframes[0] 
    for index, df in enumerate(dataframes):
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if index != 0:
            result = result.merge(df, on="Label", how = 'outer')
            result = result.loc[:, ~result.columns.str.contains('^Unnamed')]
            
            
    # result = pd.merge(df1, df2, df3, on="Label",  how = 'inner')
    # result = pd.concat([df1, df2, df3], axis=1)
    # result = result.drop(['Unnamed: 0'], axis=1)

    path = "./results/test.csv"
    result.to_csv(path)

    # Label	Precision	Recall	F-Measure	lable in Data	lable in Test Data	TP	FP	FN	TN	Test Count
