#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:17:13 2024

@author: colebarbes
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns
data = pd.read_csv('FoodTypeDataset.csv')

names = ['label'+str(i) for i in range(0, len(data.columns)-1)]
names.append('target')

print(names)

data.columns = names

print(data.head())

# Define independent (features) and dependent (targets) variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


 
# split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True, stratify=y)

print(X_train)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)

# Create an MLPClassifier instance with default parameters
mlp = MLPClassifier(hidden_layer_sizes=500, alpha=0.1, random_state=1, batch_size=10)

# Train the model on the scaled training data
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)


# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(mlp.score(X_test, y_test))

sns.heatmap(X_train, square=False, cmap=sns.color_palette('Greens'))
