import kMeans
from numpy import *
from imp import reload

# dataMat = mat(kMeans.loadDataSet('./testSet.txt'))
# kMeans.randCent(dataMat, 2)
# kMeans.distEclud(dataMat[0], dataMat[1])
# myCentroids, clustAssing = kMeans.kMeans(dataMat, 4)

datMat3 = mat(kMeans.loadDataSet('./testSet2.txt'))
centList, myNewAssments = kMeans.biKmeans(datMat3, 3)
