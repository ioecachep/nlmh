# IMPORT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# DATA
dt = pd.read_csv("abalone.data", delimiter=",");
dt.info()
dt.describe()
for label in "MFI":
    dt[label] = dt["sex"] == label
dt["age"] =  dt["rings"] + 1.5
dt['sex'] = LabelEncoder().fit_transform(dt['sex'].tolist())
new_dt = dt.copy()
new_dt['n_rings_1'] = np.where(dt.rings <= 8, 1, 0)
new_dt['n_rings_2'] = np.where(((dt.rings >8) & (dt.rings <=10)),2,0)
new_dt['n_rings_3'] = np.where(dt.rings > 10, 3, 0)
new_dt['n_rings'] = new_dt['n_rings_1'] + new_dt['n_rings_2'] + new_dt['n_rings_3']
new_dt
data = new_dt.drop(['rings','age','sex','n_rings_1','n_rings_2','n_rings_3','n_rings'],axis = 1)
data
target = new_dt.n_rings
target
# Tìm giá trị max_depth mà sẽ cho ra kết quả cao nhất 
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
find_max_depth = []
for i in range(1,100):
    model = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=i, min_samples_leaf=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)*100
    find_max_depth.append(score)

import matplotlib.pyplot as plt 
plt.scatter(range(1,100), find_max_depth)
plt.xlabel("Giá trị cần tìm")
plt.ylabel("Kết quả dự đoán")
plt.show()
# Giá trị max_depth = 6 sẽ cho ra kết quả dự đoán là cao nhất 