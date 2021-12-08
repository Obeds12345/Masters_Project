import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

dataframe = pd.read_csv('./result.csv')


X_fields = ['PDAGE', 'PDLANG', 'PDBIRTH', 'PDZIP',
            'Nearest_cancer_center_zip', 'Km_to_nearest_cancer_center',
            'Duration_to_nearest_cancer_center',
        ]

X_dataframe = dataframe[X_fields]
X_dataframe.head()

pearsoncorr = X_dataframe.corr(method='pearson')
print(pearsoncorr)