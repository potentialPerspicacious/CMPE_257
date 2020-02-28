import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

load_train = pd.read_csv('train.csv', nrows=10000)
print(load_train.shape)

y = load_train['label']
x = load_train.drop(columns='label')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

linear_model = SVC(kernel='linear', C=5, gamma=0.05)
linear_model.fit(x_train, y_train)
y_pred = linear_model.predict(x_test)
print("Accuracy for linear model: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix for linear model:", '\n', confusion_matrix(y_test, y_pred))

support_vectors_n = linear_model.n_support_
print("Support vector for each class: ", support_vectors_n)
print("Accuracy by classification_report: ", '\n', classification_report(y_test, y_pred))
#print("Precision, Recall and f_score for linear model: ", '\n', precision_recall_fscore_support(y_test, y_pred, average=None))

non_linear_model = SVC(kernel='rbf', C=5, gamma=0.05)
non_linear_model.fit(x_train, y_train)
y_nlpred = non_linear_model.predict(x_test)
print("Accuracy for non linear model: ", accuracy_score(y_test, y_nlpred))
print("Confusion Matrix for non-linear model: ", '\n', confusion_matrix(y_test, y_nlpred))

nl_support_vectors_n = non_linear_model.n_support_
print("Support vectors for each class, non-linear model", nl_support_vectors_n)
print("Accuracy by classification_report, non-linear model: ", '\n', classification_report(y_test, y_nlpred))


