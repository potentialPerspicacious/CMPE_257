import pandas as pd

data = pd.DataFrame({"ID":
["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],

                     "Marital":
["Single", "Married", "Single", "Married", "Divorced", "Married", "Divorced", "Single", "Married", "Single"],

                     "Estate":
["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],

                     "Income":
["125", "100", "70", "120", "95", "60", "220", "85", "75", "90"],

                     "Evade":
["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes"]},

                    columns=
["ID", "Marital", "Estate", "Income", "Evade"])

print(data)

data.to_csv('/Users/sughk/PycharmProjects/decision_tree/data_debtevasion.csv', index = None, header=True)



