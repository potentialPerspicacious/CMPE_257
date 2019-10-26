import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
import pydotplus

col_names = ['ID', 'Marital', 'Estate', 'Income', 'Evade']
de = pd.read_csv("/Users/sughk/PycharmProjects/decision_tree/data_debtevasion.csv", header=None, names=col_names)

de = de.iloc[1:]

de_p = preprocessing.LabelEncoder()
de["Marital"] = de_p.fit_transform(de["Marital"])
de["Estate"] = de_p.fit_transform(de["Estate"])
de["Evade"] = de_p.fit_transform(de["Evade"])

feature_cols = ["Marital", "Estate", "Income"]
target = ["Evade"]

x = de[feature_cols]
y = de[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

clf_data = DecisionTreeClassifier(criterion='entropy')
clf_data = clf_data.fit(x_train, y_train)
y_pred = clf_data.predict(x_test)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_de = tree.export_graphviz(clf_data, out_file=None,
                              feature_names=feature_cols,
                              filled=True, rounded=True,
                              special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_de)
graph.write_png('DE_tree.png')
