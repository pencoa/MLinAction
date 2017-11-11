import regTrees
from imp import reload
from numpy import *

myDat = regTrees.loadDataSet('./ex00.txt')
myMat = mat(myDat)
regTrees.createTree(myMat)
