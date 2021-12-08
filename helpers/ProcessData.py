import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MultiLabelBinarizer,Normalizer, StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from .ColumnLabelEncoder import MultiColumnLabelEncoder

class ProcessData:
    # def __init__(self):
    #     self
        
    def PrepareData():
        dataframe = pd.read_csv('result.csv')
        dataframe = dataframe[dataframe.PDZIP != 'None']
        dataframe = dataframe[dataframe.PDAGE != 'None']
        dataframe = dataframe[dataframe.PDLANG != 'None']

        normalize = [ 
            'PDBIRTH',
            'PDLANG',
            'PDZIP',
            'Km_to_nearest_cancer_center',
            'Duration_to_nearest_cancer_center',
            'Nearest_cancer_center_zip',
        ]
       
        X_fields = [
            'PDAGE', 'PDLANG', 'PDBIRTH', 'PDZIP',
            'Nearest_cancer_center_zip', 'Km_to_nearest_cancer_center',
            'Duration_to_nearest_cancer_center',
        ]
        
        LabelEncoder_fields = ['PDBIRTH']
        
        dataframe = MultiColumnLabelEncoder(columns = LabelEncoder_fields).fit_transform(dataframe)
        
        # print('mean', dataframe['PDBIRTH'].mean())
        # print('std', dataframe['PDBIRTH'].std())
        
        drops = ['IDSUBJ']
        dataframe = dataframe.drop(drops, axis=1)
        
        X = dataframe[X_fields].values
        y = dataframe.drop(X_fields, axis=1).values
        columns = dataframe.drop(X_fields, axis=1).columns

        return X, y, columns
    
    
    # send papper to wang to ask for his opinion
    # questions and answers
    # refrences
    # plagiarism