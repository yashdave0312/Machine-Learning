import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv("heart.csv")
print(df.head())

print(df.shape)
print(df.describe())
print(df.info())

#EDA 

def plots(var,num):
    plt.subplot