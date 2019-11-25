# IMPORT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
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
# ----------------------------------------- train_test_split ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
# ----------------------------------------- DecisionTreeClassifier ----------------------------------
model = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=6, min_samples_leaf=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
print("Cây quyết định DecisionTreeClassifier: ", accuracy_score(y_test, y_pred)*100, "%")
# Kết quả: 61.05873821609862 %
# ----------------------------------------- GaussianNB ----------------------------------------------
model_bayes = GaussianNB()
model_bayes.fit(X_train, y_train)
y_pred = model_bayes.predict(X_test)
print("Bayes GaussianNB: ", accuracy_score(y_test, y_pred)*100, "%")
# Kết quả: 56.92530819434373 %

# --------------------------------------------- K Fold -----------------------------------------------
kf = KFold(n_splits = 50)
