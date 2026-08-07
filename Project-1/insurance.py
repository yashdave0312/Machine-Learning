import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")
print(df)

# Exploratory Data Analysis (EDA)
# print(df.shape)
# print(df.info())
# print(df.describe())

# print(df.isnull().sum())

# numeric =["age", "bmi", "charges"]
# for col in numeric:
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[col],kde=True,bins = 20)
#     plt.show()

# categorical = ["sex", "children", "smoker", "region"]
# for col in categorical:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x = df[col])
#     plt.show()

# for col in numeric:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x = df[col])
#     plt.show()

# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

# Data Cleaning and Preprocessing
df_cleaned = df.copy()

df_cleaned.drop_duplicates(inplace=True)
print("After removing duplicates:", df_cleaned.shape)

print("Data types after cleaning:")
print(df_cleaned.dtypes)

print("Count of each category in 'sex':")
print(df_cleaned['sex'].value_counts())

df_cleaned['sex'] = df_cleaned['sex'].map({"male":0,"female":1})
print(df_cleaned.head())