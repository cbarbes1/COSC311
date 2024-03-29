#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:17:13 2024

@author: colebarbes
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns
data = pd.read_csv('FoodTypeDataset.csv')

names = ['label'+str(i) for i in range(0, len(data.columns)-1)]
names.append('target')

data.columns = names
for i in range(1, 21):
    print(i)
    print(data[data['target'] == i ].mean())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Create an MLPClassifier instance 
mlp = MLPClassifier(hidden_layer_sizes=500, solver='adam', alpha=0.05, random_state=11, batch_size=100)

# Train the model on the scaled training data
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)


# Evaluate the model
print("Confusion Matrix:")
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(mlp.score(X_test, y_test))

sns.heatmap(conf_mat, fmt='.1f', cmap='Blues')
