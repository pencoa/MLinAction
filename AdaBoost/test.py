import adaboost
from imp import reload
from numpy import *

# dataMat, classLabels = adaboost.loadSimData()
# # D = mat(ones((5, 1))/5)
# # adaboost.buildStump(dataMat, classLabels, D)
# classifierArray = adaboost.adaboostTrainDS(dataMat, classLabels, 30)
# aa = adaboost.adaClassify([[5, 5], [0, 0]], classifierArray)

dataArr, labelArr = adaboost.loadDataSet('./horseColicTraining2.txt')
classifierArray = adaboost.adaBoostTrainDS(dataArr, labelArr, 10)

testArr, testLabelArr = adaboost.loadDataSet('./horseColicTest2.txt')
prediction10 = adaboost.adaClassify(testArr, classifierArray)
errArr = mat(ones((67, 1)))
errArr[prediction10 != mat(testLabelArr).T].sum()
