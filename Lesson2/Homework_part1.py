from random import random

times=1000
hits=0
for i in range(times):
    x=random()
    y=random()
    if x*x+y*y<=1:
        hits+=1
print(4.0*hits/times)