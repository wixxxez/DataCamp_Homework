import SVMLinearIris
class Factory():

    def factory(self):
        pass

class SVMLinearFactory(Factory):

    def factory(self):

        return SVMLinearIris.SVMLinear()

class SVMRBFFactory(Factory):

    def factory(self):

        return  SVMLinearIris.SVMRBF()

class SVMPolyFactory(Factory):

    def factory(self):

        return SVMLinearIris.SVMPolynom()
