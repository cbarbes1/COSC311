from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set() 

# function to find best number of components
def PCA_find(X):
    for i in range(1, 65):
        pca = PCA(n_components = i)
        X_new = pca.fit_transform(X)
        cumulative_variance = sum(pca.explained_variance_ratio_)
        if cumulative_variance >= 0.85:
            return i

# load the dataset
digits = load_digits()

X = digits['data']
y = digits['target']

# scale the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# do the analysis
num_components = PCA_find(X)
print(f'Number of components to keep at least 85% variance: {num_components}')

# get the actual components
pca = PCA(n_components=num_components)
X_new = pca.fit_transform(X)
X_new = pca.transform(X)
 # kmean cluster the dataset
kmeans = KMeans(n_clusters=10, random_state=42, algorithm='elkan')
clusters = kmeans.fit_predict(X_new)
print("Centers of the Clusters: ")
print(kmeans.cluster_centers_)

labels = np.zeros_like(clusters)
for i in range(64):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]


print(accuracy_score(digits.target, labels))

mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()