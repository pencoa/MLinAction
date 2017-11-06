from imp import reload
import svmMLiA

# dataArr, labelArr = svmMLiA.loadDataSet('./testSet.txt')
# b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, ('lin', 1))
#
# svmMLiA.testRbf(k1=0.2)
svmMLiA.testDigits(('rbf', 20))
