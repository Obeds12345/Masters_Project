import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

# url = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv'
# df = pd.read_csv(url, index_col=0)
df = pd.read_csv("test.csv")
df.columns = df.columns.str.replace(".", "_")
df.head()


maov = MANOVA.from_formula('F_1 + F_2 + F_3 ~ Label', data=df)

print(maov.mv_test())