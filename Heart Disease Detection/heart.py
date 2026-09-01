import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr,chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
import joblib

warnings.filterwarnings("ignore")

df = pd.read_csv("heart.csv")
print(df.head())

# print(df.shape)
# print(df.describe())
# print(df.columns)
# print(df.describe())

#EDA 

print(df.duplicated().sum())
print(df["HeartDisease"].value_counts())

print(df.isnull().sum())

# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde=True,bins=20)
#     plt.show()

# plotting("Age",1)
# plotting("RestingBP",2)
# plotting("Cholesterol",3)
# plotting("MaxHR",4)
# plt.tight_layout()

ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0,ch_mean)
print(df["Cholesterol"].head())

restbp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
df["RestingBP"] = df["RestingBP"].replace(0,restbp_mean)

categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# for col in categorical:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x = df[col], hue = df["HeartDisease"])
#     plt.show()

# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

# Data  Preprocessing
df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)
print(df_encoded.head())

df_encoded = df_encoded.astype(int)
print(df_encoded.head())

# scalarisation
scalar = StandardScaler()
scale_cols = ["Age","RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

df_encoded[scale_cols] = scalar.fit_transform(df_encoded[scale_cols])
print(df_encoded.head())

# print(df_encoded.columns)

# sns.heatmap(df_encoded.corr(numeric_only=True), annot=True)
# plt.show()

# Pearson Correlation - used to find relation between target and input variables

selected_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP',
       'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST',
       'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
correlations = {
    feature: pearsonr(df_encoded[feature], df_encoded["HeartDisease"])[0] 
    for feature in selected_features       
}

correlations_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Pearson Correlation"])
correlations_df = correlations_df.sort_values(by="Pearson Correlation", ascending=False)
print(correlations_df)

# chi2 test - used to find relation between target and input variables

categorical_encoded = [ 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP',
       'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST',
       'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

alpha = 0.05
list = []
for cols in categorical_encoded:
    contingency_table = pd.crosstab(df_encoded[cols], df_encoded["HeartDisease"])
    chi2, p, _ , _ = chi2_contingency(contingency_table)
    print(f"Chi-square test for {cols}:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p}")
    if p < alpha:
        print(f"Reject the null hypothesis: {cols} is associated with charges.")
        list.append(cols)
    else:
        print(f"Fail to reject the null hypothesis: {cols} is not associated with charges.")

print(list)

df_final = df_encoded[['Age','RestingBP','Cholesterol', 'FastingBS', 'MaxHR','Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up','HeartDisease']]
print(df_final)

X = df_final.drop("HeartDisease",axis = 1)
y = df_final["HeartDisease"]
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Logistic Regression" : LogisticRegression(),
    "KNN Classification" : KNeighborsClassifier(n_neighbors = 5),
    "Naive Bayes Classification" : GaussianNB(),
    "Support Vector Machines" : SVC(),
    "Decision Tree " : DecisionTreeClassifier()
}

result = []

for name,model in models.items() :
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    result.append({
        "model" : name,
        "accuracy" : round(accuracy,2),
        "f1_score" : round(f1,2)
    })

print(result)


# joblib.dump(models["KNN Classification"],"model_knn.pkl")
# joblib.dump(scalar,"scalar.pkl")
# joblib.dump(X.columns.tolist(),"columns.pkl")