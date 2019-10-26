import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

i_load = datasets.load_iris()
print(i_load.DESCR)
i_data = pd.DataFrame(i_load.data)
feature_names = i_load.feature_names
target_names = i_load.target_names
print("Feature names of iris data set is", feature_names, '\n')
print("Targets names of iris data set is", '\n', target_names, '\n')
# print(i_data)
x = i_data[i_data.columns[0:2]]
print("First two feature columns are: ", '\n', x)
x_min, x_max = x[0].min() - 0.8, x[0].max() + 0.6
y_min, y_max = x[1].min() - 0.5, x[1].max() + 0.6
# print(x_min, x_max, y_min, y_max)
plt.scatter(x[0], x[1], c=i_load.target, cmap=plt.cm.Set1)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()

target_values = i_load.target
# print(target_values)
for i in range(0, len(target_values)):
    if target_values[i] == 0:
        target_values[i] = 1
    else:
        target_values[i] = 0

print("Y:Labels for sepal is: ", '\n', target_values)
x_train, x_test, y_train, y_test = train_test_split(x, target_values, test_size=0.2, random_state=3)
lr_clf = LogisticRegression(random_state=3, solver='lbfgs')
lr_clf.fit(x_train, y_train)
y_prob = lr_clf.predict_proba(x_test)
y_pred = lr_clf.predict(x_test)
print("Score: ", lr_clf.score(x_test, y_test))
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Estimated predicted probabilities are as follows:", '\n', y_prob)
cvs = cross_val_score(lr_clf, x_train, y_train, cv=4)
print("Cross validation score for logistic regression", cvs)

#print(y_train.shape)
#print(target_values.shape)
#print(x_train.shape)
#print(x.shape)

kf = KFold(n_splits=4)

for train_index, test_index in kf.split(x_train, y_train):
    print("K-fold_TRAIN: ", train_index, '\n', "K-fold_TEST: ", test_index)
    x_ktrain, x_ktest = x_train.iloc[train_index], x_train.iloc[test_index]
    y_ktrain, y_ktest = y_train[train_index], y_train[test_index]

lr_clf.fit(x_ktrain, y_ktrain)
print("Accuracy for K_fold is: ", lr_clf.score(x_ktest, y_ktest))
cvs_kf = cross_val_score(lr_clf, x_ktrain, y_ktrain, cv=kf)
print('\n', "Cross validation score for kfold is: ", cvs_kf, '\n')

skf = StratifiedKFold(n_splits=4)

for train_i, test_i in skf.split(x_train, y_train):
    print("SK-fold_TRAIN: ", train_i, '\n', "SK-fold_TEST: ", test_i)
    x_sktrain, x_sktest = x_train.iloc[train_index], x_train.iloc[test_index]
    y_sktrain, y_sktest = y_train[train_index], y_train[test_index]

print("Accuracy for K_fold is: ", lr_clf.score(x_sktest, y_sktest))
cvs_skf = cross_val_score(lr_clf, x_sktrain, y_sktrain, cv=skf)
print('\n', "Cross validation score for stratifiedKfold is: ", cvs_skf)
