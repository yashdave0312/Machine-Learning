import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr,chi2_contingency

warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")
print(df)

# Exploratory Data Analysis (EDA)
print(df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())

numeric =["age", "bmi", "charges"]
for col in numeric:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True,bins = 20)
    plt.show()

categorical = ["sex", "children", "smoker", "region"]
for col in categorical:
    plt.figure(figsize=(6,4))
    sns.countplot(x = df[col])
    plt.show()

for col in numeric:
    plt.figure(figsize=(6,4))
    sns.boxplot(x = df[col])
    plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

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

df_cleaned['smoker'] = df_cleaned['smoker'].map({"no":0,"yes":1})
print(df_cleaned.head())

df_cleaned.rename(columns = 
    {"sex" : "isfemale",
   "smoker" : "issmoker"}, inplace = True)

df_cleaned = pd.get_dummies(df_cleaned,columns = ["region"], drop_first = True)
print(df_cleaned.head())

df_cleaned = df_cleaned.astype(int)
print("Data types after conversion to int:")
print(df_cleaned.dtypes)
print(df_cleaned.head())

#  Featture Engineering and Extraction

sns.histplot(df_cleaned["bmi"],kde=True,bins =20)
plt.show()

df_cleaned["bmi_category"] = pd.cut(
    df_cleaned["bmi"],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=["Underweight", "Normal weight", "Overweight", "Obesity"]
)

print(df_cleaned.head())
df_cleaned = pd.get_dummies(df_cleaned,columns = ["bmi_category"], drop_first= True)
print(df_cleaned.head())

df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())

# Feature Scaling

cols = ["age", "bmi", "children"]
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

# Pearson Correlation - used to find relation between target and input variables

selected_features = ["age", "bmi", "children", "isfemale", "issmoker", "region_northwest", "region_southeast", "region_southwest","bmi_category_Normal weight", "bmi_category_Overweight", "bmi_category_Obesity"]
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned["charges"])[0] 
    for feature in selected_features       
}

correlations_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Pearson Correlation"])
correlations_df = correlations_df.sort_values(by="Pearson Correlation", ascending=False)
print(correlations_df)

cat_features = ["isfemale", "issmoker", "region_northwest", "region_southeast", "region_southwest","bmi_category_Normal weight", "bmi_category_Overweight", "bmi_category_Obesity"]
alpha = 0.05
df_cleaned["charges_bins"] = pd.qcut(df_cleaned["charges"], q=4, labels=False)
print(df_cleaned.head())

for cols in cat_features:
    contingency_table = pd.crosstab(df_cleaned[cols], df_cleaned["charges_bins"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test for {cols}:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p}")
    if p < alpha:
        print(f"Reject the null hypothesis: {cols} is associated with charges.")
    else:
        print(f"Fail to reject the null hypothesis: {cols} is not associated with charges.")

final_df = df_cleaned[["age","isfemale","bmi","children","issmoker","charges","region_southeast","bmi_category_Obesity"]]
print(final_df)