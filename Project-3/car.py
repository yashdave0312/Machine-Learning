import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr,chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

df = pd.read_csv("Car_prediction.csv")

# EDA

# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

# print(df.drop_duplicates())

# print(df.isnull().sum())
# print(df.dtypes)

numerical = ["year","price","mileage","tax","mpg","engineSize"]

# for cols in numerical :
#     sns.histplot(x = df[cols],kde = True)
#     plt.show()

print(df.loc[df["engineSize"] == 0,"engineSize"].count())
print(df.columns)

categorical = ["model","transmission","fuelType"]
# for cols in categorical:
#     sns.countplot(x = df[cols])
#     plt.show()

print(df.loc[df["fuelType"] == "Electric","fuelType"].count())

print(df["model"].value_counts())

# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

# Data Preprocessing started

print(categorical)
print(df["fuelType"].value_counts())
print(df["transmission"].value_counts())

df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)
print(df_encoded.head())

df_encoded = df_encoded.astype(int)
print(df_encoded.head())

# Scalarisation

scaler = StandardScaler()
scal_cols =["year","mileage","tax","mpg","engineSize"]

df_encoded[scal_cols] = scaler.fit_transform(df_encoded[scal_cols])
print(df_encoded.head())


# Pearson Correlation
print(df_encoded.columns)
selected_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'model_ C-MAX',
       'model_ EcoSport', 'model_ Edge', 'model_ Escort', 'model_ Fiesta',
       'model_ Focus', 'model_ Fusion', 'model_ Galaxy', 'model_ Grand C-MAX',
       'model_ Grand Tourneo Connect', 'model_ KA', 'model_ Ka+',
       'model_ Kuga', 'model_ Mondeo', 'model_ Mustang', 'model_ Puma',
       'model_ Ranger', 'model_ S-MAX', 'model_ Streetka',
       'model_ Tourneo Connect', 'model_ Tourneo Custom',
       'model_ Transit Tourneo', 'model_Focus', 'transmission_Manual',
       'transmission_Semi-Auto', 'fuelType_Electric', 'fuelType_Hybrid',
       'fuelType_Other', 'fuelType_Petrol']

correlations = {
    feature: pearsonr(df_encoded[feature], df_encoded["price"])[0] 
    for feature in selected_features       
}

correlations_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Pearson Correlation"])
correlations_df = correlations_df.sort_values(by="Pearson Correlation", ascending=False)
print(correlations_df)

# Chi2 test

new_categorical =  ['model_ C-MAX',
       'model_ EcoSport', 'model_ Edge', 'model_ Escort', 'model_ Fiesta',
       'model_ Focus', 'model_ Fusion', 'model_ Galaxy', 'model_ Grand C-MAX',
       'model_ Grand Tourneo Connect', 'model_ KA', 'model_ Ka+',
       'model_ Kuga', 'model_ Mondeo', 'model_ Mustang', 'model_ Puma',
       'model_ Ranger', 'model_ S-MAX', 'model_ Streetka',
       'model_ Tourneo Connect', 'model_ Tourneo Custom',
       'model_ Transit Tourneo', 'model_Focus', 'transmission_Manual',
       'transmission_Semi-Auto', 'fuelType_Electric', 'fuelType_Hybrid',
       'fuelType_Other', 'fuelType_Petrol']

# sns.histplot(x = df_encoded["price"],kde=True)
# plt.show()

alpha = 0.05
df_encoded["price_bins"] = pd.qcut(df_encoded["price"], q=4, labels=False)
print(df_encoded.head())
list = []
for cols in new_categorical:
    contingency_table = pd.crosstab(df_encoded[cols], df_encoded["price_bins"])
    chi2, p, _ , _ = chi2_contingency(contingency_table)
    print(f"Chi-square test for {cols}:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p}")
    if p < alpha:
        print(f"Reject the null hypothesis: {cols} is associated with charges.")
        list.append(cols)
    else:
        print(f"Fail to reject the null hypothesis: {cols} is not associated with charges.")

print(list)        

final_df = df_encoded[['year','price','mileage','tax','mpg','engineSize','model_ C-MAX', 'model_ EcoSport', 'model_ Edge', 'model_ Fiesta', 'model_ Focus', 'model_ Fusion', 'model_ Galaxy', 'model_ Grand C-MAX', 'model_ Grand Tourneo Connect', 'model_ KA', 'model_ Ka+', 'model_ Kuga', 'model_ Mustang', 'model_ Puma', 'model_ S-MAX', 'model_ Tourneo Connect', 'model_ Tourneo Custom', 'transmission_Manual', 'transmission_Semi-Auto', 'fuelType_Hybrid', 'fuelType_Petrol']]
print(final_df.shape)
print(final_df.head())


# Model Training 

X = final_df.drop("price",axis =1)
y = final_df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
print(model)

y_pred = model.predict(X_test)
print(y_pred)

r2 = r2_score(y_test,y_pred)
print(final_df.shape)

n = X_test.shape[0]
P = X_test.shape[1]

adjusted_r2 = 1- ((1 - r2)*(n-1) / (n - P - 1))

print(r2)
print(adjusted_r2) 
