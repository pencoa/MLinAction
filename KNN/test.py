import kNN
from imp import reload
from numpy import *

group, labels = kNN.createDataSet()
kNN.classify0([0, 0], group, labels, 3)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')


import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()


normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
kNN.handwritingClassTest()
