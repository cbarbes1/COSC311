import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def hand_written_digits():
    digits = load_digits()
    clf = svm.SVC(kernel='rbf', C=10, gamma=0.001)
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    # split training and test set
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.20, random_state=42, shuffle=True, stratify=digits.target)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.score(X_test, y_test))
    print("Confusion Matrix:")
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
    
def cluster_GPS_data():
    with open('housing.csv', 'r') as file:
        data = pd.read_csv(file)

    plt.scatter(data['longitude'].values, data['latitude'].values, s=50)

if __name__ == "__main__":
    choice = input("Enter which program you would like to run: ")
    if(choice == "Digits"):
        hand_written_digits()
    elif(choice == "Housing"):
        cluster_GPS_data()