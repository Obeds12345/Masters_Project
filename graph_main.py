import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('helpers'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt

def plotData(dfs, classifiers, file_):
    Labels = [
        'CM',
        'LI1',
        'I1',
        'F1',
        'SP1',
        'L2',
        'LI2',
        'F2',
        'I7',
        'FP5',
        'I6',
        'MM7',
        'OTHER',
        'FP1',
        'CP4',
        'D2',
        'CC4',
        'MM1',
        'MM5',
        # 'main',
        'CP5',
        'MM6',
        'CP4',
        'F3',
        'I8',
        'SP4',
        'T3',
        'L1',
        'NB1',
        'I5',
        'W1',
        'CP1',
        'FP2',
        'MM9',
        'FP6',
        'PB1',
        'CP2',
        'I3',
        'AP3',
        'L12',
    ]
    
    graphs = [
        ['Label','Precision'],
        ['Label','Recall'],
        ['Label','F-Measure'],
        ['Label','True Positive'],
        ['Label','False Positive'],
        ['Label','False Negative'],
        ['Label','True Negative'],
    ]

    
    fig, axs = plt.subplots(7, 1, figsize=(15,45))
    fig.suptitle('Classifications')
    fig.subplots_adjust(hspace=0.5)
    
    for graph, df in zip(range(len(graphs)), dfs):
        df = df[df['Label'].isin(Labels)]
        # df = df[df.Precision != 0]
        for classifier in classifiers:
            axs[graph].plot(df["Label"], df[classifier])
        
        axs[graph].xaxis.set_tick_params(rotation=90)
        axs[graph].grid()
        axs[graph].legend(classifiers)
       
        title = graphs[graph][1]
        
        axs[graph].set_ylabel(title, fontsize=25)
        axs[graph].set_xlabel("Determinants of Health", fontsize=25)
    # plt.show()
    plt.savefig('./results/classification_comparison_{}.pdf'.format(file_)) 
    print("*********************..........DONE ..........*********************")
    
    
def generateDataframe(columns, dataframes, headers_):
    dfs = []
    
    for column in columns:
        data = []
            
        for index, df in enumerate(dataframes):
            if index == 0:
                data.append(df["Label"])
            data.append(df[column])
        
        headers = ["Label", *headers_]
        df = pd.concat(data, axis=1, keys=headers)
        dfs.append(df)
    return dfs

 
if __name__=='__main__':
    # df1 = pd.read_csv('./results/custom/fold_average/BinaryRelevance(GaussianNB()).csv')
    # df2 = pd.read_csv('./results/custom/fold_average/ClassifierChain(GaussianNB()).csv')
    # df3 = pd.read_csv('./results/custom/fold_average/OneVsRestClassifier(KNeighborsClassifier()).csv')
    
    # df1 = pd.read_csv('results/custom/fold_average/BinaryRelevance(GaussianNB()).csv')
    # df2 = pd.read_csv('results/custom/fold_average/BinaryRelevance(KNeighborsClassifier()).csv')
    # df3 = pd.read_csv('results/custom/fold_average/BinaryRelevance(RandomForestClassifier()).csv')

    # df4 = pd.read_csv('results/custom/fold_average/ClassifierChain(GaussianNB()).csv')
    # df5 = pd.read_csv('results/custom/fold_average/ClassifierChain(KNeighborsClassifier()).csv')
    # df6 = pd.read_csv('results/custom/fold_average/ClassifierChain(RandomForestClassifier()).csv')

    df7 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(GaussianNB()).csv')
    df8 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(KNeighborsClassifier()).csv')
    df9 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(RandomForestClassifier()).csv')
    df10 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(SVC()).csv')
    
    # df11 = pd.read_csv('results/custom/fold_average/CNN.csv')

    # df4 = pd.read_csv('./results/custom/fold_average/CNN.csv')
    
    columns = ['Precision', 'Recall', 'F-Measure', 'TP', 'FP', 'FN', 'TN']
    classifiers = [
        # "BinaryRelevance_GaussianNB", "BinaryRelevance_KNeighborsClassifier", "BinaryRelevance_RandomForestClassifier",
        # "ClassifierChain_GaussianNB", "ClassifierChain_KNeighborsClassifier", "ClassifierChain_RandomForestClassifier",
        "OneVsRestClassifier_GaussianNB", "OneVsRestClassifier_KNeighborsClassifier", "OneVsRestClassifier_RandomForestClassifier",
        "SVC_GaussianNB", 
        # "CNN", 
        ]
    
    dataframes = [  df7, df8, df9, df10,]
    # dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]
    dataframes_ = []
    for index, df in enumerate(dataframes):
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if index != 0:
            target_df = dataframes[0]
            df = df.set_index('Label')
            df = df.reindex(index=target_df['Label'])
            df = df.reset_index()
        dataframes_.append(df)
        
    # for index, df in enumerate(dataframes_):
        # print(df.head())
                   
    dfs = generateDataframe(columns, dataframes_, classifiers)
    # file_ = 
    plotData(dfs, classifiers, "OneVsRestClassifier")