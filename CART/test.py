import regTrees
from imp import reload
from numpy import *

# myDat = regTrees.loadDataSet('./ex00.txt')
# myMat = mat(myDat)
# regTrees.createTree(myMat)

# myDat1 = regTrees.loadDataSet('./ex0.txt')
# myMat1 = mat(myDat1)
# regTrees.createTree(myMat1)

# myDat2 = regTrees.loadDataSet('./ex2.txt')
# myMat2 = mat(myDat2)
# myTree = regTrees.createTree(myMat2, ops=(0, 1))
# myDatTest = regTrees.loadDataSet('./ex2test.txt')
# myMat2Test = mat(myDatTest)
# myTreeTest = regTrees.prune(myTree, myMat2Test)

myMat2 = mat(regTrees.loadDataSet('./exp2.txt'))
regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, (1, 10))
