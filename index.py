import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

dt = pd.read_csv("data.csv", delimiter=",")
data = dt.iloc[:,:30]
target = dt.Result

# ----------------------------------------- train_test_split ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1/3, random_state=5)

model_tree = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=15, min_samples_leaf=1)
model_tree.fit(X_train, y_train)
y_pred = model_tree.predict(X_test)
print("Cây quyết định (train_test_split): ", round(accuracy_score(y_test, y_pred)*100, 2), "%")
# Kết quả: 95.82%

model_bayes = GaussianNB()
model_bayes.fit(X_train, y_train)
y_pred = model_bayes.predict(X_test)
print("Bayes (train_test_split): ", round(accuracy_score(y_test, y_pred)*100, 2), "%")
# Kết quả: 58.62%

# --------------------------------------------- K Fold -----------------------------------------------
kf = KFold(n_splits = 50)

model_tree = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=15, min_samples_leaf=1)
scores = []
for train_index, test_index in kf.split(data):
	X_train, X_test = data.iloc[train_index,], data.iloc[test_index,]
	y_train, y_test = target.iloc[train_index,], target.iloc[test_index]
	model_tree.fit(X_train, y_train)
	y_pred = model_tree.predict(X_test)
	score = accuracy_score(y_test, y_pred)
	scores.append(score)
total = 0
for i in range(len(scores)):
    total += scores[i]
result = total/len(scores)
print("Cây quyết định (K Fold): ", round(result * 100, 2), "%")
# Kết quả trung bình: 96.25%

model_bayes = GaussianNB()
scores = []
for train_index, test_index in kf.split(data):
	X_train, X_test = data.iloc[train_index,], data.iloc[test_index,]
	y_train, y_test = target.iloc[train_index,], target.iloc[test_index]
	model_bayes.fit(X_train, y_train)
	y_pred = model_bayes.predict(X_test)
	score = accuracy_score(y_test, y_pred)
	scores.append(score)
total = 0
for i in range(len(scores)):
    total += scores[i]
result = total/len(scores)
print("Bayes (K Fold):", round(result * 100, 2), "%")
# Kết quả trung bình: 60.31%