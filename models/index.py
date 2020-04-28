from models.logisticReg import log_reg
from models.svm import svm
from models.random_forest import random_forest
from models.ada_boost import ada_boost
from models.naiive import naiive
from .knn import knn
from sklearn.neighbors import KNeighborsClassifier
def model(x, y, mod="adaboost-forest"):
    if mod == 'ada-naiive':
        return naiive(x,y)
    elif mod == "svm":
        return svm(x, y)
    elif mod == 'knn':
        return knn(x,y)
    elif mod == "log-reg":
        return log_reg(x, y)
    elif mod == "random_forest":
        return random_forest(x, y)
    elif mod == "ada-dt":
        return ada_boost(x, y, "dt")
    elif mod == 'ada-rf':
        return ada_boost(x, y, 'rf')
    elif mod == 'ada-svm':
        return ada_boost(x, y, 'svm')
    elif mod == 'ada-log':
        return ada_boost(x,y,'log')
    else:
        print("DEFAULT")
