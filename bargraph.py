import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

def barPlot(df, title):
    # sns.set_theme(style="whitegrid")
    # ax = sns.barplot(x="algorithm", y="F_Measure", data=df)
    # ax = sns.boxplot(x='algorithm', y='F_Measure', data=df, color='#99c2a2')
    # ax = sns.swarmplot(x="algorithm", y="F_Measure", data=df, color='#7d0013')
    # df.plot.bar(x='algorithm', y='F_Measure')
    # print(df[df[‘Name’]==’Donna’].index.values)
    # print("The Precision is {:.2f}".format(df["Precision"].mean()) )
    # print("The Recall is {:.2f}".format(df["Recall"].mean()) )
    # print("The F_Measure is {:.2f}".format(df["F_Measure"].mean()) )
    df = df.sort_values(by=['F_Measure'], ascending=False)
    maxLabel = df.iloc[[0]]
    b_a = maxLabel['algorithm'].values[0]
    b_a_p = maxLabel["Precision"].mean()
    b_a_r = maxLabel["Recall"].mean()
    b_a_f_m = maxLabel["F_Measure"].mean()
    # print("The Precision is {:.2f}".format(maxLabel["Precision"].mean()) )
    # print("The Recall is {:.2f}".format(maxLabel["Recall"].mean()) )
    # print("The F_Measure is {:.2f}".format(maxLabel["F_Measure"].mean()) )
    # print(best_algorithm)
    # print(df.iloc[[0]])
    # print(df.iloc[[0]])
    # column = df["F_Measure"]
    # max_value = column.max()
    # print(max_value)
    # print(df[['F_Measure']].idxmax())
    # print(df.loc[df['F_Measure'] == max_value])
    # print(df.loc[df['F_Measure'].isin("{}".format(max_value))])
    # print(df[df["F_Measure"]== "max_value"].index.values)
    

    df.plot(x="algorithm", y=["Precision", "Recall", "F_Measure"], kind="bar")
    plt.title("Label {} => Precsion {:.2f}, Recall {:.2f}, F_Measure {:.2f}, Classifier {}".format(title,b_a_p,b_a_r,b_a_f_m,b_a ))
    # plt.title("Label {}, {} ".format(title, best_algorithm))
    plt.xticks(rotation = 45)
    # plt.show()
    plt.savefig('./results/boxplots/{}.png'.format(title))   


summary = pd.read_csv('results/summary_fold_average.csv')
summary = summary[summary.F_Measure != 0]
summary = summary[summary.Recall != 0]
summary = summary[summary.Precision != 0]

barries = summary.loc[:,'Label'].to_numpy()
barries = np.unique(barries)

# print(df.describe())
for barry in barries:
    df = summary.loc[summary['Label'] == barry]
    if len(df) > 3:
        print('Label ', barry)
        barPlot(df, barry)
        print()
