import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv("heart.csv")
print(df.head())

print(df.shape)
print(df.describe())
print(df.columns)
print(df.describe())

#EDA 

print(df.duplicated().sum())
print(df["HeartDisease"].value_counts())

print(df.isnull().sum())

def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True,bins=20)
    plt.show()

plotting("Age",1)
plotting("RestingBP",2)
plotting("Cholesterol",3)
plotting("MaxHR",4)
plt.tight_layout()

ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0,ch_mean)
print(df["Cholesterol"].describe())