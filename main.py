#region Includes
import pandas as pd
from utils import *

from preprocess.index import pre_process
#endregion

p("READING THE DATA... ")
#region Reading Data
train = pd.read_csv('heart_train.csv')
test = pd.read_csv('heart_test.csv')
del train["index"]
train.dropna(inplace=True)
desc = train.describe()
p(desc)
#endregion
delim()

p("COLUMN SELECTION...")
#region column selection
x_train = train.loc[:, train.columns != "target"]
y_train = train.loc[:, train.columns == "target"]
sample_size = 2
print(f"""
Final Training data:
    Shape:
        x_train: {x_train.shape}
        y_train: {y_train.shape}
    Sample:
        x_train: \n{x_train[:sample_size]}
        y_train: \n{y_train[:sample_size]}
""")
#endregion
delim()

p("PRE-PROCESSING...")
#region pre-processing
x_train = pre_process(x_train)
p(x_train[:2])
#endregion
delim()



p("MODEL TRAINING...")
#region models
#endregion
delim()

p("PREDICTION...")
#region predection
#endregion
delim()

p("PLOTTING...")
#region accuracy & plotting
#endregion