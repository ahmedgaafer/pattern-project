from models.logisticReg import log_reg
from models.svm import svm
from models.random_forest import random_forest
from models.ada_boost import ada_boost

def model(x, y, mod="adaboost-forest"):

    if mod == "svm":
        return svm(x, y)
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
    else:
        print("DEFAULT")
