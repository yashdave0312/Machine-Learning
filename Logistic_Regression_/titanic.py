import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings 

warnings.filterwarnings("ignore")

df = sns.load_dataset("titanic")
print(df.head())
print(df.shape)

print(df.columns)

df.drop(["who","adult_male","alive","embark_town","deck","class"],axis =1,inplace = True)
print(df.head())

print(df.isnull().sum())

df["age"] = df["age"].fillna(df["age"].mean(),inplace = True)
print(df["age"].isnull().sum())

