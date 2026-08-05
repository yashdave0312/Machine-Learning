import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway

df = sns.load_dataset("titanic")
print(df)

print(df['pclass'].unique())

class_1 = df[df['pclass']==1]['age']
class_2 = df[df['pclass']==2]['age']
class_3 = df[df['pclass']==3]['age']

print("Class 1 ages:\n", class_1)
print("Class 2 ages:\n", class_2)
print("Class 3 ages:\n", class_3)

f_statistic, p_value = f_oneway(class_1.dropna(), class_2.dropna(), class_3.dropna())

print("F-statistic:", f_statistic)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("I will reject the null hypothesis.")
else:
    print("I will accept the null hypothesis.")    