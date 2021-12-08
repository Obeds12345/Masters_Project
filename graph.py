import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('helpers'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def plotData(dfs):
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
        'main',
        'CP5'
    ]
    graphs = [
         ['Label','Precision'],
        ['Label','Recall'],
        ['Label','True Positive'],
        ['Label','False Positive'],
        ['Label','False Negative'],
        ['Label','True Negative'],
        ['Label','Count'],
    ]

    
    fig, axs = plt.subplots(7, 1, figsize=(15,45))
    fig.suptitle('CNN Thresholds')
    fig.subplots_adjust(hspace=1)
    
    for graph, df in zip(range(len(graphs)), dfs):
        # df = df[df["0.3"] != 0 | df['Label3'].isin(Labels)] 
        df = df[df['Label'].isin(Labels)] 
        
        # axs[graph].bar(df["Label3"], df["0.3"], 0.2, label='0.3')
        # axs[graph].bar(df["Label4"], df["0.4"], 0.2, label='0.4')
        # axs[graph].bar(df["Label5"], df["0.5"], 0.2, label='0.5')
        axs[graph].plot(df["Label"], df["0.3"])
        axs[graph].plot(df["Label"], df["0.4"])
        axs[graph].plot(df["Label"], df["0.5"])
        axs[graph].plot(df["Label"], df["0.6"])
        axs[graph].plot(df["Label"], df["0.7"])
        axs[graph].plot(df["Label"], df["0.8"])
        axs[graph].xaxis.set_tick_params(rotation=90)
        axs[graph].grid()
        # axs[graph].legend(["0.3", "0.4", "0.5"])
        axs[graph].legend(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
        title = graphs[graph][1]
        
        axs[graph].set_ylabel(title, fontsize=25)
        axs[graph].set_xlabel("Determinants of Health", fontsize=25)
    # plt.show()
    plt.savefig('./results/CNN_CV/Thresholds_comparison.pdf')   
    
def generateDataframe(columns, dataframe3,dataframe4,dataframe5,dataframe6,dataframe7,dataframe8):
    dfs = []
    for column in columns:
        data = [
            dataframe3["Label"], 
            dataframe3[column], 
            dataframe4[column],
            dataframe5[column], 
            dataframe6[column],
            dataframe7[column],
            dataframe8[column],
        ]
        headers = ["Label", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
        df = pd.concat(data, axis=1, keys=headers)
        dfs.append(df)
    return dfs
    
 
if __name__=='__main__':
    dataframe3 = pd.read_csv('./results/CNN_CV/0.3.csv')
    dataframe4 = pd.read_csv('./results/CNN_CV/0.4.csv')
    dataframe5 = pd.read_csv('./results/CNN_CV/0.5.csv')
    dataframe6 = pd.read_csv('./results/CNN_CV/0.6.csv')
    dataframe7 = pd.read_csv('./results/CNN_CV/0.7.csv')
    dataframe8 = pd.read_csv('./results/CNN_CV/0.8.csv')
    # for dta in [dataframe3,dataframe4,dataframe5,dataframe6,dataframe7,dataframe8]:
    #     dta = dta.sort_values(by=['Label'], ascending=False)
    #     dta = dta.reset_index(drop=True)    
    
    
    dataframe4 = dataframe4.set_index('Label')
    dataframe4 = dataframe4.reindex(index=dataframe3['Label'])
    dataframe4 = dataframe4.reset_index()
    
    dataframe5 = dataframe5.set_index('Label')
    dataframe5 = dataframe5.reindex(index=dataframe3['Label'])
    dataframe5 = dataframe5.reset_index()
    
    dataframe6 = dataframe6.set_index('Label')
    dataframe6 = dataframe6.reindex(index=dataframe3['Label'])
    dataframe6 = dataframe6.reset_index()
    
    dataframe7 = dataframe7.set_index('Label')
    dataframe7 = dataframe7.reindex(index=dataframe3['Label'])
    dataframe7 = dataframe7.reset_index()
    
    dataframe8 = dataframe8.set_index('Label')
    dataframe8 = dataframe8.reindex(index=dataframe3['Label'])
    dataframe8 = dataframe8.reset_index()
       
    # dataframe3 = dataframe3.sort_values(by=['Label'], ascending=False)
    # dataframe3 = dataframe3.reset_index(drop=True)    
    
    # dataframe4 = dataframe4.sort_values(by=['Label'], ascending=False)
    # dataframe4 = dataframe4.reset_index(drop=True)    

    # dataframe5 = dataframe5.sort_values(by=['Label'], ascending=False)
    # dataframe5 = dataframe5.reset_index(drop=True)    
    
    # dataframe6 = dataframe6.sort_values(by=['Label'], ascending=False)
    # dataframe6 = dataframe6.reset_index(drop=True) 

    # dataframe7 = dataframe5.sort_values(by=['Label'], ascending=False)
    # dataframe7 = dataframe5.reset_index(drop=True) 
    
    # dataframe8 = dataframe8.sort_values(by=['Label'], ascending=False)
    # dataframe8 = dataframe8.reset_index(drop=True) 

    
    columns = ['Precision', 'Recall', 'TP', 'FP', 'FN', 'TN', 'Count']
    dfs = generateDataframe(columns, dataframe3,dataframe4,dataframe5,dataframe6,dataframe7,dataframe8)
    plotData(dfs)
    # data = [
    #     dataframe3["Label"], 
    #     dataframe3["Precision"], 
    #     dataframe4["Precision"],
    #     dataframe5["Precision"], 
    #     dataframe6["Precision"],
    #     dataframe7["Precision"],
    #     dataframe8["Precision"]
    #     ]
    # headers = ["Label", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
    # df = pd.concat(data, axis=1, keys=headers)


    # fig, axs = plt.subplots(6, 1, figsize=(15,45))
    # fig.suptitle('CNN Thresholds')
    # fig.subplots_adjust(hspace=1)
    
    # axs[0].plot(df["Label"], df["0.3"])
    # axs[0].plot(df["Label"], df["0.4"])
    # axs[0].plot(df["Label"], df["0.5"])
    # axs[0].plot(df["Label"], df["0.6"])
    # axs[0].plot(df["Label"], df["0.7"])
    # axs[0].plot(df["Label"], df["0.8"])
    # axs[0].xaxis.set_tick_params(rotation=90)
    # axs[0].grid()
    # axs[0].legend(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
    # axs[0].set_ylabel("Thresholds", fontsize=25)
    # axs[0].set_xlabel("Label", fontsize=25)
    # # plt.show()
    # plt.savefig('./results/CNN_CV/all.pdf')    
    
    