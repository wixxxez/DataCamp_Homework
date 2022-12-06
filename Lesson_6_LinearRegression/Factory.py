import Ridge
import LinearRegression
import Lasso

class Factory :

    def factory(self):

        pass;

class RidgeFactory(Factory):

    def factory(self):

        return Ridge.RidgeRegression()

class LinearRegFactory(Factory):

    def factory(self):

        return LinearRegression.LinearRegress()

class LassFactory(Factory):

    def factory(self):

        return Lasso.LassoRegression()