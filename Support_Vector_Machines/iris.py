import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

df = sns.load_dataset("iris")
print(df.head())

X = df.drop("species",axis = 1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# KNN model testing by grid search cv

model_knn = KNeighborsClassifier(n_neighbors = 9)

model_knn.fit(X_train,y_train)

y_pred = model_knn.predict(X_test)
print(y_test)
print(y_pred)

print("KNN score",model_knn.score(X_test,y_test))
# now here the score is coming 100% , that means , my model is overfitiing
# now lets use the grid search cv for knn

classifier2 = GridSearchCV((model_knn),{
    "weights" : ["uniform", "distance"],
    "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
},cv= 5,return_train_score=False)

classifier2.fit(X,y)
results1 = pd.DataFrame(classifier2.cv_results_)
print(results1[["param_weights","param_algorithm","mean_test_score"]])


# svm model testing by grid search cv

model_svm = SVC(gamma="auto")

model_svm.fit(X_train,y_train)
y_pred1 = model_svm.predict(X_test)
print("SVM score : ",model_svm.score(X_test,y_test))

# now lets use the grid search cv for svm

classifier = GridSearchCV((model_svm),{
    "C" : [1,10,20,30],
    "kernel" : ["linear","rbf"]
},cv= 5,return_train_score=False)

print(classifier)

classifier.fit(X,y)
print(classifier.cv_results_)

results = pd.DataFrame(classifier.cv_results_)
print(results[["param_C","param_kernel","mean_test_score"]])