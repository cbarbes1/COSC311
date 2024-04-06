import pandas as pd
import stats
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def plot_result(model, X, y):
    means = np.mean(X, axis=1)  # Calculate the mean of each row
    for item, type, mean_value in zip(X, y, means):
        y_pred = model.predict([item])
        #print("Room # ", y_pred, " actual: ", type)
        plt.scatter(y_pred, mean_value)
        #plt.text(y_pred, mean_value, fontsize=9, ha='center')

    plt.xlabel('Class')
    plt.ylabel('Mean of Row')
    plt.title('Class vs. Mean of Row')
    plt.show()

data = pd.read_csv('wifi_localization.txt', header=None, sep="\t")
columns=["label"+str(i) for i in range(0, len(data.columns)-1)]
columns.append('target')
data.columns = columns

data.plot()

for i in range(1, 6):
    print("Label ", i," data correlation: ", stats.correlation(xs=data.iloc[:, i-1],ys=data.iloc[:, i]))

# Define independent (features) and dependent (targets) variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


 
# split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


print(X_train)


for i in range(0, 7):
    plt.scatter(X[:, i], X[:, i+1 if i < 6 else 0], s=50)

neighbors = np.arange(1, 25)
accuracy = np.empty(len(neighbors))
 
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy[i] = knn.score(X_train, y_train)

plt.figure(figsize=(5, 3))
plt.title('k-NN self-test accuracy')
plt.plot(neighbors, accuracy)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# KNN classification

knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(X_train, y_train)

y_predict = knn.predict(X_train)

conf_mat = confusion_matrix(y_train, y_predict)
print(conf_mat)
print(classification_report(y_train, y_predict))
knn.score(X_train, y_train)
sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


plot_result(knn, X_train, y_train)



DT = DecisionTreeClassifier()
tree = DT.fit(X_train, y_train)
y_pred = DT.predict(X_train)

conf_mat = confusion_matrix(y_train, y_pred)
print(conf_mat)
print(classification_report(y_train, y_pred))
DT.score(X_train, y_train)

DT.score(X_train, y_train)
sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

plot_result(DT, X_train, y_train)


# Define independent (features) and dependent (targets) variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


 
# split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

print(X_train)

neighbors = np.arange(1, 25)
accuracy = np.empty(len(neighbors))
 
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy[i] = knn.score(X_test, y_test)

plt.figure(figsize=(5, 3))
plt.title('k-NN self-test accuracy')
plt.plot(neighbors, accuracy)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# KNN classification

knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

conf_mat = confusion_matrix(y_test, y_predict)
print(conf_mat)
print(classification_report(y_test, y_predict))
knn.score(X_test, y_test)
sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()



plot_result(knn, X_test, y_test)

DT = DecisionTreeClassifier(class_weight='balanced', max_features=3)
tree = DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(classification_report(y_test, y_pred))
DT.score(X_test, y_test)

sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

plot_result(DT, X_train, y_train)

# KNN classification

knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

conf_mat = confusion_matrix(y_test, y_predict)
print(conf_mat)
print(classification_report(y_test, y_predict))
knn.score(X_test, y_test)
sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

accuracies = []
for i in np.arange(0.1, 0.60, 0.10):
    # split training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    # KNN classification

    knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
    knn.fit(X_train, y_train)

    y_predict = knn.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_predict)
    print(conf_mat)
    print(classification_report(y_test, y_predict))
    accuracies.append(knn.score(X_test, y_test))
    sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


plt.plot(['10%', '20%', '30%', '40%', '50%'], accuracies)