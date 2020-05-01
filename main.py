# region Includes
import pandas as pd
from utils import *
from models.index import model
from preprocess.index import pre_process
import matplotlib.pyplot as plt
# endregion

p("READING THE DATA... ")
# region Reading Data
train = pd.read_csv('heart_train.csv')
test = pd.read_csv('heart_test.csv')
y_test = pd.read_csv('sample.csv')

del train["index"]
del test["Index"]
del y_test["Index"]


train.dropna(inplace=True)
desc = train.describe()
p(desc)
# endregion
delim()

p("COLUMN SELECTION...")
# region column selection
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
# endregion
delim()

p("PRE-PROCESSING...")
# region pre-processing

x_train, y_test, y_train = pre_process(x_train, test, y_train)
# endregion
delim()


p("MODEL TRAINING...")
# region models
mod = 'log-reg'
clf = model(x_train, y_train, mod)

# endregion
delim()

p("EXPORTING RESULTS TO CSV ...")
# region export

pr = clf.predict(y_test)
f = {'Index': [i for i in range(243, 304)], "target": pr}
df = pd.DataFrame(f)
df.to_csv('ans.csv', index=False)
# endregion
