import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X,y_true = make_blobs(n_samples=500,centers=3,cluster_std=0.60,random_state=42)

df = pd.DataFrame(X,columns=["Feature_1","Feature_2"])
print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(X_scaled)