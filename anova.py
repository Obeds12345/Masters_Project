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

def boxplot(df, title):
    sns.set_theme(style="whitegrid")
    # ax = sns.barplot(x="algorithm", y="F_Measure", data=df)
    ax = sns.boxplot(x='algorithm', y='F_Measure', data=df, color='#99c2a2')
    ax = sns.swarmplot(x="algorithm", y="F_Measure", data=df, color='#7d0013')
    # df.plot.bar(x='algorithm', y='F_Measure')
    plt.title("Label {}".format(title))
    plt.xticks(rotation = 45)
    # plt.show()
    plt.savefig('./results/boxplots/{}.pdf'.format(title))   


summary = pd.read_csv('results/summary_fold.csv')
summary = summary[summary.F_Measure != 0]
summary = summary[summary.Recall != 0]
summary = summary[summary.Precision != 0]

barries = summary.loc[:,'Label'].to_numpy()
barries = np.unique(barries)
# barries = ['LI1']

# print(df.describe())
for barry in barries:
    df = summary.loc[summary['Label'] == barry]
    if len(df) > 5:
        print('Label ', barry)
        mod = ols('F_Measure ~ label_algorithm', data=df).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        print(aov_table)
        
        # perform multiple pairwise comparison (Tukey HSD)
        m_comp = pairwise_tukeyhsd(endog=df['F_Measure'], groups=df['label_algorithm'], alpha=0.05)
        print(m_comp)
        # boxplot(df, barry)
        print()
        print()


# print('Label')
# mod = ols('F_Measure ~ Label', data=summary).fit()
# aov_table = sm.stats.anova_lm(mod, typ=2)
# print(aov_table)

# print('Algorithm')
# mod = ols('F_Measure ~ algorithm', data=summary).fit()
# aov_table = sm.stats.anova_lm(mod, typ=2)
# print(aov_table)

# print( 1.104405e-49 > 0.05)
# print( 0.439378 > 0.05)

# ANOVA results with combinations of 2 groups:
# formula = 'F_Measure ~ C(Label) + C(algorithm) + C(Label):C(algorithm) + C(algorithm):C(Label)'
# formula = 'F_Measure ~ Label + algorithm'
# lm = ols(formula, summary).fit()
# table = sm.stats.anova_lm(lm, typ=2)
# print(table)


# perform multiple pairwise comparison (Tukey HSD)
# m_comp = pairwise_tukeyhsd(endog=summary['Recall'], groups=summary['algorithm'], alpha=0.05)
# print(m_comp)