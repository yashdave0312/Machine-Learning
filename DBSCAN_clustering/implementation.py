import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler

X,y_true = make_moons(n_samples=500 , noise = 0.05,random_state=42)
# print(X)
# print(y_true)

df = pd.DataFrame(X,columns =["Feature_1","Feature_2"])
print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(df)

# Finding that which clustering algorithm is better , kmeans or dbscan

# Kmeans 

wcss = []
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(k_range,wcss,marker = "o")
plt.show()

kmeans_final = KMeans(n_clusters=6,random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df["clusters"] = cluster_labels

sns.scatterplot(x = df["Feature_1"],y= df["Feature_2"],hue=df["clusters"],palette="tab10")
plt.show()

# DBSCAN

dbscan = DBSCAN(eps=0.3,min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

df["dbscan_clusters"] = dbscan_labels

sns.scatterplot(x = df["Feature_1"],y=df["Feature_2"],hue=df["dbscan_clusters"])
plt.show()