import adaboost
from imp import reload
from numpy import *

dataMat, classLabels = adaboost.loadSimData()
# D = mat(ones((5, 1))/5)
# adaboost.buildStump(dataMat, classLabels, D)
classifierArray = adaboost.adaboostTrainDS(dataMat, classLabels, 9)
