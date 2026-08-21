import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# EDA
df = pd.read_csv("customer.csv")
print(df.head())

print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())

df["Review Rating"] = df["Review Rating"].fillna(df["Review Rating"].mean(),inplace = True)
print(df.isnull().sum())

print(df.dtypes)

df = df.drop("Customer ID",axis=1)
print(df.head())

print(df["Item Purchased"].value_counts())
df = df.drop("Item Purchased",axis=1)
print(df.head())

numeric = []
categorical=[]

# for col in df.columns:
#     if df[col].dtype == "int64":
#         numeric.append(col)
#     elif df[col].dtype == "float":
#         numeric.append(col)
#     else :
#         categorical.append(col)

# for col in numeric :
#     sns.histplot(x = df[col],kde=True)
#     plt.show()

# for col in categorical:
#     sns.countplot(x= df[col])
#     plt.show()

# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()

print(df.columns)
# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()
# plt.tight_layout()

for col in categorical:
    print(df[col].value_counts())

df = df.drop(["Color","Location"],axis=1)
print(df.head())
# print(df.columns)

for col in df.columns:
    if df[col].dtype == "int64":
        numeric.append(col)
    elif df[col].dtype == "float":
        numeric.append(col)
    else :
        categorical.append(col)

# print(categorical)

# Data Preprocessing

print(df["Subscription Status"].value_counts())
df = pd.get_dummies(df,columns = ['Gender', 'Category', 'Size', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases'],drop_first=True)
print(df.columns)

df = df.astype("int")
print(df.head())

print(numeric)

# Standard Scalarisataion

scaler = StandardScaler()
df[numeric] = scaler.fit_transform(df[numeric])
print(df.head())
