import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings 
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

warnings.filterwarnings("ignore")

df = sns.load_dataset("titanic")
# print(df.head())
# print(df.shape)

# print(df.columns)

df.drop(["who","adult_male","alive","embark_town","deck","class"],axis =1,inplace = True)
# print(df.head())

# print(df.isnull().sum())

df["age"] = df["age"].fillna(df["age"].mean(),inplace = True)
# print(df["age"].isnull().sum())

df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0],inplace = True)
print(df.isnull().sum())

# you have only two null values in the embarked column , so you can drop the rows as well
# df.dropna(subset=["embarked"],inplace = True)

print(df.head())

# Data Preprocessing

df["sex"] = df["sex"].map({"male":0,"female":1})
print(df.head())

print(df["embarked"].value_counts())
df = pd.get_dummies(df,columns = ["embarked"], drop_first= True)
print(df.head())

df = df.astype("int")
print(df.head())

# Feature Scaler

scaler = StandardScaler()
scal_cols = ["pclass", "age","fare","sibsp","parch"]
print(df["sibsp"].value_counts())
print(df["parch"].value_counts())

df[scal_cols] = scaler.fit_transform(df[scal_cols])
print(df.head())

# sns.heatmap(df.corr(numeric_only=True),annot = True)
# plt.show()

df.drop("embarked_Q", axis=1, inplace=True)
print(df.head())

final_df = df[[ "pclass","sex","age","sibsp","parch","fare","alone","embarked_S","survived"]]
print(final_df.head())

X = final_df.drop("survived",axis = 1)
y = final_df["survived"]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = SVC(kernel = "rbf")

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

conf = confusion_matrix(y_test,y_pred)
print(conf)

print(classification_report(y_test,y_pred))

scores = cross_val_score(model,X,y,cv = 5,scoring = "accuracy")
print(scores)

print(scores.mean())

if (scores.mean() > accuracy):
    print("Cross validation is succesfully implemented not normal accuray score.")
else:
    print("Normal accuracy method is implemented , not cross validation.")    