import DataSet
import SVMFactory

iris = DataSet.IrisDataSet()

X = iris.Load2Features()
Y = iris.getY()

labels_dict = iris.getTargetNames()

def start(factory: SVMFactory.Factory, n_f: int = 1 ):
    svm = factory.factory()
    svm.setXY(X, Y)
    svm.model()
    svm.printScore()
    if n_f == 2:
        svm.plotBoundary(labels_dict)

# Use n_f to plot decision boundary if you use 2 features
print("SVM Linear Kernel")

start(SVMFactory.SVMLinearFactory(), n_f=2)

print("SVM RBF Kernel")

start(SVMFactory.SVMRBFFactory(), n_f=2)

print("SVM Polynomial Kernel")

start(SVMFactory.SVMPolyFactory(), n_f=2)