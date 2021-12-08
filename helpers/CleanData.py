import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MultiLabelBinarizer,Normalizer, StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from ColumnLabelEncoder import MultiColumnLabelEncoder

# class ProcessData:
    # def __init__(self):
    #     self
        
def PrepareData():
    dataframe = pd.read_csv('data.csv')
    
    dataframe['target'] = dataframe['R24Barrier']
    dataframe['target'] = dataframe['target'].apply(lambda x: x.upper())
    dataframe['target'] = dataframe['target'].apply(lambda x: re.sub("-[0-9]*","", x))
    # dataframe['R24Barrier'] = dataframe['R24Barrier'].apply(lambda x: re.sub("NONE","", x))
    # dataframe['R24Barrier'] = dataframe['R24Barrier'].apply(lambda x: re.sub("NONE[0-9]*","", x))
    dataframe['target'] = dataframe['target'].apply(lambda x: re.sub("&","", x))
    dataframe['target'] = dataframe['target'].apply(lambda x: x.split( ))
    dataframe['Nearest_cancer_center_zip'] = dataframe['Nearest_cancer_center'].str.extract(r'(\d{5}\-?\d{0,4})')
  
       
    multilabel = MultiLabelBinarizer()
    target = multilabel.fit_transform(dataframe['target'])
    
    
    drops = ['Generated_at', 'Nearest_cancer_center',
            'Nearest_hospital','Km_to_nearest_hospital','target', 'R24Barrier','R24Action',
        ]
    
    dataframe = dataframe.drop(drops, axis=1)
    
    target_df = pd.DataFrame(target, columns=multilabel.classes_)
    target_df = target_df.drop(['6', 'NONE'], axis=1)
    # print(target_df.info())
    result = pd.concat([dataframe, target_df], axis=1)
    
    result = result[result.PDZIP != 'None']
    result = result[result.PDAGE != 'None']
    result = result[result.PDLANG != 'None']
    
    dataframe = dataframe[dataframe.Km_to_nearest_cancer_center != 'None']
    
    result['Duration_to_nearest_cancer_center'] = result['Duration_to_nearest_cancer_center'].apply(lambda x: x.replace(" mins", ""))
    result['Km_to_nearest_cancer_center'] = result['Km_to_nearest_cancer_center'].apply(lambda x: x.replace("km", ""))
    result['Km_to_nearest_cancer_center'] = result['Km_to_nearest_cancer_center'].apply(lambda x: x.replace(",", ""))
    result['Km_to_nearest_cancer_center'] = result['Km_to_nearest_cancer_center'].apply(lambda x: x.strip())
    result['Km_to_nearest_cancer_center'] = result['Km_to_nearest_cancer_center'].apply(lambda x: float(x))
    
    result["Km_to_nearest_cancer_center"] = pd.to_numeric(result["Km_to_nearest_cancer_center"])
    result["Duration_to_nearest_cancer_center"] = pd.to_numeric(result["Duration_to_nearest_cancer_center"])
    
    result.to_csv(r"./result.csv", index=False)
        
if __name__=='__main__': 
    PrepareData()
    # 60166 convert Duration_to_nearest_cancer_center to mins