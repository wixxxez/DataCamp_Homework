import random
a = random.randint(1, 100);
b=random.randint(1, 100);
randomProd = a*b
randomSum = a+b

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

    def FirstStage(self, number):
        prod_parts = []

        for i in range(1, number + 1):
            if (number % i == 0):
                prod_parts.append(i)

        self.prodParts = prod_parts;

        return (len(prod_parts)>2)
    def FindProd(self, combi):

        for i in combi:
            if(i[0]*i[1]==self.prod):
                return i;
def check_pairs(Prod_obj, combi):
    true_combi = []
    for i in combi:
        if(Prod_obj.FirstStage(i[0]) and Prod_obj.FirstStage(i[1])):
            true_combi.append(i);
    return true_combi
print("Prod is: ", randomProd)
print("Sum is: ", randomSum)

Prod_s = Prod(randomProd)
Prod_s.FirstStage(randomProd)
print("1st step: ",Prod_s.prodParts)
Suma_s = Suma(randomSum)
combi = Suma_s.getAllCombinations();

print("2nd step: ",combi)
combi = check_pairs(Prod_s,combi)
print("3rd part: ", combi );

print("Last step: ", Prod_s.FindProd(combi))