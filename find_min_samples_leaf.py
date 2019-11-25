import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = pd.read_csv("data.csv", delimiter=",")
data = dt.iloc[:,:30]
target = dt.Result

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1/3, random_state=5)

# Tìm giá trị min_samples_leaf mà sẽ cho ra kết quả cao nhất
find_min_samples_leaf = []
for i in range(1,100):
    model = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=15, min_samples_leaf=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)*100
    find_min_samples_leaf.append(score)

import matplotlib.pyplot as plt 
plt.scatter(range(1,100), find_min_samples_leaf)
plt.xlabel("Giá trị cần tìm")
plt.ylabel("Kết quả dự đoán")
plt.show()
# Giá trị min_sample_leaf = 1 sẽ cho ra kết quả dự đoán là cao nhất 