from sklearn.preprocessing import PolynomialFeatures

class Polynomial:

    def getScaledData(self, x_train, x_test, degree):

        poly = PolynomialFeatures(degree=2, include_bias=False)
        x_poly_train = poly.fit_transform(x_train)
        x_poly_test = poly.transform(x_test)
        return x_poly_train,x_poly_test;