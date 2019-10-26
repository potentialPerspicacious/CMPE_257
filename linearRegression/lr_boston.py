import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

b_data = load_boston()
print("\n")
print("Dictionary keys are:",
      b_data.keys())  # dictionary keys are 'data' , 'target' , 'feature_names', 'DESCR', 'filename'
print("\n")
print("Description about", '\n', b_data.DESCR)  # 506 instances, 13 attributes, median value $14 is target
print("\n")
print("Data set size is:", b_data.data.shape)  # It has 506 rows and 13 columns
print("Feature names are:",
      b_data.feature_names)  # 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
print("Size of target vector is:", b_data.target.shape)  # 506 rows, matches with data shape
print("\n")

bos_data = pd.DataFrame(b_data.data)
bos_data.columns = b_data.feature_names
bos_data["Price"] = b_data.target

# x = bos_data.drop("Price", axis=1)
# y = bos_data["Price"]


feature_cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
res = ["Price"]
x = bos_data[feature_cols]
y = bos_data[res]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
print("Size of training data_features:", x_train.shape)
print("Size of test data_features:", x_test.shape)
print("Size of training data_target:", y_train.shape)
print("Size of test data_target", y_test.shape)
print("\n")
bos_lr = LinearRegression()
bos_lr.fit(x_train, y_train)
y_pred = bos_lr.predict(x_test)
print(y_pred.shape)
print("Model report are as follows")
bos_mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error is:", bos_mse)
bos_r2 = r2_score(y_test, y_pred)
print("R2 Score is:", bos_r2)
slope = bos_lr.coef_
slp = np.transpose(slope)
print("Slopes of each attributes is as follows", '\n',  slp)
print(slope.shape)
print(pd.DataFrame({"Attributes": feature_cols}))
#print(pd.DataFrame({"Coefficients": slope}))
# print(x[:1].columns)
#t1 = pd.DataFrame({"Attributes": x.columns, "Coefficients": slope})
# t1 = pd.DataFrame({"Attributes"[x.columns]}, columns="Attributes")
#print(t1)
# slopeTabel = pd.DataFrame(slope, columns=['coefficients'])
# print("The slope of Regression line w.r.t each attribute is as follows: ", '\n', slopeTabel)
intercept = bos_lr.intercept_
print("The intercept of the Regression Line is: ", intercept)
