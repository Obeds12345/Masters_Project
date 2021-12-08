import pandas as pd
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

athleisure_df = pd.read_csv('test.csv')

print('F_measure')
mod = ols('F_measure ~ algorithm', data=athleisure_df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)



# perform multiple pairwise comparison (Tukey HSD)
m_comp = pairwise_tukeyhsd(endog=athleisure_df['F_measure'], groups=athleisure_df['algorithm'], alpha=0.05)
print(m_comp)