from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from configs import *


def ada_boost(x, y, mod="dt"):
    if mod == 'dt':
        clf = DecisionTreeClassifier(random_state=randomstate, max_depth=3)
    elif mod == 'rf':
        clf = RandomForestClassifier(random_state=randomstate, max_depth=3)
    elif mod == 'svm':
        clf = SVC(random_state=43)
    else:
        clf = DecisionTreeClassifier(random_state=randomstate, max_depth=3)

    estimator = 200
    btd = AdaBoostClassifier(clf,
                             algorithm="SAMME",
                             n_estimators=estimator, random_state=randomstate
                             )
    btd.fit(x, y)

    return btd
