from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

data = load_breast_cancer()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy of the DT with depth 3 is: ", metrics.accuracy_score(y_test, y_pred))

clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=50)

clf_ada.fit(x_train, y_train)
yhat_pred = clf_ada.predict(x_test)

print("Accuracy with Adaboost is: ", metrics.accuracy_score(y_test, yhat_pred))

clf_Mada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                              n_estimators=500,
                              learning_rate=0.1)
clf_Mada.fit(x_train, y_train)
yhatM_pred = clf_Mada.predict(x_test)

print("Accuracy of Adaboost with l_R = 0.1 is: ", metrics.accuracy_score(y_test, yhatM_pred))
