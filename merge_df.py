import pandas as pd

df1 = pd.read_csv('results/custom/fold_average/BinaryRelevance(GaussianNB()).csv')
df2 = pd.read_csv('results/custom/fold_average/BinaryRelevance(KNeighborsClassifier()).csv')
df3 = pd.read_csv('results/custom/fold_average/BinaryRelevance(RandomForestClassifier()).csv')

df4 = pd.read_csv('results/custom/fold_average/ClassifierChain(GaussianNB()).csv')
df5 = pd.read_csv('results/custom/fold_average/ClassifierChain(KNeighborsClassifier()).csv')
df6 = pd.read_csv('results/custom/fold_average/ClassifierChain(RandomForestClassifier()).csv')

df7 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(GaussianNB()).csv')
df8 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(KNeighborsClassifier()).csv')
df9 = pd.read_csv('results/custom/fold_average/OneVsRestClassifier(RandomForestClassifier()).csv')
df10 = pd.read_csv('results/custom/fold_average/CNN.csv')

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]

algorithm = [
            'BR_NB', 'BR_KN', 'BR_RF', 
            'CC_NB', 'CC_KN', 'CC_RF',  
            'OVR_NB', 'OVR_KN', 'OVR_RF',  
            'CNN'
            ]

for index, df in enumerate(frames):
    df['algorithm'] = pd.Series([algorithm[index] for x in range(len(df.index))])
    df["label_algorithm"] = df["algorithm"] + "_" + df["Label"]
# frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]

result = pd.concat(frames)
result.rename(columns={'F-Measure': 'F_Measure'}, inplace=True)
# result["F_Measure"] = result["F-Measure"]
# result = result.drop(['F-Measure'], axis=1)

result = result[result.F_Measure != 0]
result = result[result.Recall != 0]
result = result[result.Precision != 0]
# result = result.reset_index(drop=True)    

path = "./results/summary_fold_average.csv"

result.to_csv(path)
