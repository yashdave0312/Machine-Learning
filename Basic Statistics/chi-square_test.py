import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
print(df)

contingency_table = pd.crosstab(df['sex'],df['survived'])
print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-square statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)

alpha = 0.05

if p < alpha:
    print("I will reject the null hypothesis.") 
else:
    print("I will accept the null hypothesis.")
    