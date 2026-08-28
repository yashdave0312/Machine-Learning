import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X,y = make_blobs(n_samples=500,n_features=5,centers=3,cluster_std=1.5,random_state=42)

X_df = pd.DataFrame(X , columns = ["Feature 1","Feature 2","Feature 3","Feature 4","Feature 5"])
print(X_df.head())

scaler =StandardScaler()
X_df_scaled = scaler.fit_transform(X_df)
print(X_df_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_df_scaled)
print(X_pca)

df_pca = pd.DataFrame(X_pca , columns = ["PC1","PC2"])