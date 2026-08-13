import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("Car_prediction.csv")
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

print(df.drop_duplicates())

print(df.isnull().sum())
print(df.dtypes)

numerical = ["year","price","mileage","tax","mpg","engineSize"]