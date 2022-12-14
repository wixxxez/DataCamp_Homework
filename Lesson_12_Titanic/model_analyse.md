Для вирішення задачі з пасажирмами титаніка будемо використовувати деревоподібні моделі

Розглянемо: 

- Decision Tree
- Gradient Boost
- RandomForest
- XGBoost

# Decision Tree

    Decision Tree: 
    Depth:  3
    train accuracy= 81.886%
    test accuracy= 82.960%
    Depth:  5
    train accuracy= 83.533%
    test accuracy= 80.269%
    Depth:  7
    train accuracy= 85.928%
    test accuracy= 81.166%
    Depth:  9
    train accuracy= 89.521%
    test accuracy= 82.960%
    Depth:  11
    train accuracy= 92.515%
    test accuracy= 82.511%
    Depth:  13
    train accuracy= 94.760%
    test accuracy= 80.717%
    Depth:  15
    train accuracy= 96.557%
    test accuracy= 77.130%

# Gradient boost

    Depth:  3
    Learning rate:  0.01
    train accuracy= 83.533%
    test accuracy= 83.408%
    
    Depth:  5
    Learning rate:  0.01
    train accuracy= 87.275%
    test accuracy= 85.202%
    
    Depth:  7
    Learning rate:  0.01
    train accuracy= 91.317%
    test accuracy= 85.202%
    
    Depth:  9
    Learning rate:  0.01
    train accuracy= 94.760%
    test accuracy= 80.717%
    
    Depth:  11
    Learning rate:  0.01
    train accuracy= 96.856%
    test accuracy= 83.408%
    
    Depth:  13
    Learning rate:  0.01
    train accuracy= 97.754%
    test accuracy= 83.408%

# Random Forest 

    Depth:  3
    train accuracy= 82.635%
    test accuracy= 82.960%
    Depth:  5
    train accuracy= 85.778%
    test accuracy= 84.305%
    Depth:  7
    train accuracy= 89.072%
    test accuracy= 83.857%
    Depth:  9
    train accuracy= 93.263%
    test accuracy= 85.650%
    Depth:  11
    train accuracy= 96.407%
    test accuracy= 83.857%
    Depth:  13
    train accuracy= 97.754%
    test accuracy= 82.063%
    Depth:  15
    train accuracy= 98.503%
    test accuracy= 82.960%
    Depth:  17
    train accuracy= 98.503%
    test accuracy= 81.614%

# XGBoost
    train accuracy= 97.605%
    test accuracy= 80.269%

При змінні лише стандартних параметрів найкраще себе показали моделі
RandomForest i GradientBoost



