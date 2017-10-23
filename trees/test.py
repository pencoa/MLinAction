from imp import reload
import trees

myDat, labels = trees.createDataSet()
trees.calcShannonEnt(myDat)
myTree = trees.createTree(myDat, labels)

import treePlotter
treePlotter.createPlot()
