import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

df = sns.load_dataset("iris")
print(df.head())

X = df.drop("species",axis = 1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model_knn = KNeighborsClassifier(n_neighbors = 5)

model_knn(X_train,y_train)

