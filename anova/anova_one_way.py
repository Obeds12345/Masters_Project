import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

data = pd.read_csv("test.csv")

# data.boxplot('weight', by='group')
# plt.show()
print('F_1')
mod = ols('F_1 ~ Label', data=data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
print()

print('F_2')
mod = ols('F_2 ~ Label', data=data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
print()

print('F_3')
mod = ols('F_3 ~ Label', data=data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
print()




#########################################



import pandas as pd
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

athleisure_df = pd.read_csv('test.csv')

keys = list(athleisure_df.Label.unique())
# ['LI1', 'K2', 'I8', 'CM2', 'CM']
print(keys) 
values = []
for Label in list(athleisure_df.Label.unique()):
    values.append(list(athleisure_df.loc[athleisure_df['Label'] == Label, 'F_1']))
# print(values)

data = dict(zip(keys, values))
# print(data)


# # stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(data['LI1'],
                                data['K2'], 
                                data['I8'],  
                                data['CM2'], 
                                data['CM'],)