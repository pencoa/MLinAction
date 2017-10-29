from imp import reload
import logRegres
from numpy import *

dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr, labelMat)
logRegres.plotBestFit(weights.getA())

weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights)
