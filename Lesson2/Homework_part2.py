import random

randomA = random.randint(1, 100);
randomB = random.randint(1, 100);

class Suma:

    def __init__(self, suma):
        self.sum = suma

    def getAllCombinations(self):
        combinations = []
        for i in range(round(self.sum/2)+1):
            combinations.append(tuple([self.sum - i, i]))
        return combinations;

class Prod:

    def __init__(self,prod):
        self.prod = prod

    def getNumbers(self, numbers):

        for i in numbers:

            if(i[0]*i[1] == self.prod):

                return tuple([i[0],i[1]]);

SumaScientist = Suma(randomA+randomB)
ProdScientist = Prod(randomB*randomA)
print(randomA,randomB)
print("Sum is: ", randomB+randomA)
print("Prod is: ", randomB*randomA)
SumaKnowNumbers = SumaScientist.getAllCombinations()
print(ProdScientist.getNumbers(SumaKnowNumbers))