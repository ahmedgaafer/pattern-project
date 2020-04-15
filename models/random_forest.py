from sklearn.ensemble import RandomForestClassifier
from configs import *


def random_forest(x, y):
    clf = RandomForestClassifier(random_state=randomstate, max_depth=4)
    clf.fit(x, y)
    return clf
