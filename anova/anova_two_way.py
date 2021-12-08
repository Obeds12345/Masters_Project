import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")

# ANOVA
formula = 'F_1~C(Label)+F_2~C(Label)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2)

print(aov_table)