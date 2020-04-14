#region Includes
import pandas as pd

#endregion

#region Reading Data
train = pd.read_csv('heart_train.csv').to_numpy()
test = pd.read_csv('heart_test.csv').to_numpy()
x_train = train[1:14]
y_train = train[14]

print(f""" 
Data shape:
    x_train: {x_train.shape}
    y_train: {y_train.shape}
Data sample:
    x_train: {x_train[0:2,:]}
    y_train: {y_train[0:2,:]}
""")

#endregion

#region pre-processing

#endregion

#region models
#endregion

#region predection
#endregion

#region accuracy & plotting
#endregion