import pandas as pd
import stats as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,  classification_report
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


names = ['COUGH', 'DRINK', 'EAT', 'READ', 'SIT', 'WALK']
col_names = ['timestamp', 'X', 'Y', 'Z']

# define function to test a window size and offset

def sliding_window(df_list: list[pd.DataFrame], window_size, offset, output_file):
    window_df = pd.DataFrame(columns=['time_past', 'mean_X', 'mean_Y', 'mean_Z', 'mean_distance', 'target'])
    for i in range(0, len(df_list)):
        rows, columns = df_list[i].shape
        for j in range(0, rows-window_size+1, offset):
            window = df_list[i].iloc[j:j+window_size]
            new_row_df = pd.DataFrame([[window['timestamp'].to_numpy()[-1]-window['timestamp'].to_numpy()[0], window['X'].mean(), window['Y'].mean(), window['Z'].mean(), window['distance'].mean(), window['target'].to_numpy()[0]]], columns=window_df.columns)
            window_df = pd.concat([window_df, new_row_df], ignore_index=True)
            
    return window_df
            
            
# distance calculation
def distance(df: pd.DataFrame):
    return math.sqrt(float(df['X'])**2+float(df['Y'])**2 + float(df['Z'])**2)
    

# open the files into dataframes
frame_list = [pd.read_csv(name+'.csv') for name in names]

# create the data sets
for i in range(0, len(frame_list)):
    frame_list[i] = frame_list[i].iloc[1:, :]
    frame_list[i].reset_index(drop=True, inplace=True)
    frame_list[i].columns = col_names
    frame_list[i]['distance'] = frame_list[i].apply(distance, axis=1)
    frame_list[i]['target'] = len(frame_list[i])*[i]
    frame_list[i]['target'] = [item+1 for item in frame_list[i]['target']]
    frame_list[i] = frame_list[i].astype(float)

# test each window size
for i in range(100, 1024, 100):
    window_df = sliding_window(frame_list, i, 16, "test.csv")

    X = window_df.iloc[:, :-1]
    y = window_df['target']
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    print("Accuracy Score for the ",i ,"-th sliding window size")
    print(knn.score(X, y))

# fix the targets
print([len(window_df[window_df['target'] == i]) for i in range(1, 7)])

# get the best sliding window and start testing
window_df = sliding_window(frame_list, 800, 16, "test.csv")
print(window_df)
X = window_df.iloc[:, :-1]
y = window_df['target']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print(knn.score(X, y))

k = 10
 
result = cross_val_score(knn, X, y, cv = k)
 
print("Avg accuracy: {}".format(result.mean()))

normalize(X, norm="l1")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print(knn.score(X, y))
y_pred = knn.predict(X)
 
cm = confusion_matrix(y,y_pred)
print(cm)


k = 10
result = cross_val_score(knn, X, y, cv = k)
 
print("Avg accuracy: {}".format(result.mean()))


X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("Decision Tree testing")

#self test
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print(knn.score(X, y))
y_pred = knn.predict(X)
 
cm = confusion_matrix(y ,y_pred)
print(cm)
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
print(classification_report(y, y_pred))

# independent test
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
y_pred = knn.predict(X_test)
 
cm = confusion_matrix(y_test ,y_pred)
print(cm)
print(classification_report(y_test, y_pred))
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# cross validation of knn
k = 10
 
result = cross_val_score(knn, X, y, cv = k)
result_y_pred = cross_val_predict(knn, X, y)

print("Avg accuracy: {}".format(result.mean()))

print(classification_report(y, result_y_pred))
cm = confusion_matrix(y , result_y_pred)
print(cm)
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

print("Decision Tree testing")

DT = DecisionTreeClassifier()
tree = DT.fit(X, y)
y_pred = DT.predict(X)
print(DT.score(X, y))

print(classification_report(y, y_pred))

DT = DecisionTreeClassifier()
tree = DT.fit(X_train, y_train)
y_pred = DT.predict(X_train)
DT.score(X_test, y_test)

k = 10
 
result = cross_val_score(DT, X, y, cv = k)
 
print("Avg accuracy: {}".format(result.mean()))


print("Random Forest testing")
RF = RandomForestClassifier()
y_fit = RF.fit(X, y)
y_pred = RF.predict(X)
print(RF.score(X, y))
cm = confusion_matrix(y ,y_pred)
print(cm)
print(classification_report(y, y_pred))

RF = RandomForestClassifier()
y_fit = RF.fit(X, y)
y_pred = RF.predict(X)
print(RF.score(X, y))

k = 10
kf = KFold(n_splits=k, random_state=None)
 
result = cross_val_score(RF, X, y, cv = kf)

print("Avg accuracy: {}".format(result.mean()))
 