import unittest
import numpy as np
from CustomLinearReg import  Linear_Regression_1
from CustomLinearReg import  Linear_Regression_2
from CustomLinearReg import Linear_Regression_3
from CustomLinearReg import Linear_Regression_4
class H_test(unittest.TestCase):

    def test_value(self):
        # DON'T_CHANGE_THIS_CODE. It is used to let you check the result is correct
        np.random.seed(2018)
        b_check = np.random.randn()
        w_check = np.random.randn(1, 1)
        X_check = np.random.randn(10, 1)

        lin_reg_1 = Linear_Regression_1()

        self.assertEqual(lin_reg_1.h(b_check, w_check, X_check).tolist(),[
             [ 0.97328067],
             [-1.02123839],
             [ 0.01548272],
             [ 0.22131391],
             [-0.35985014],
             [-0.21271821],
             [-0.67711878],
             [-0.0244979 ],
             [ 0.02010501],
             [-0.37284922]
            ]
)
        return 0;
class J_test(unittest.TestCase):

    def test_value(self):
        # DON'T_CHANGE_THIS_CODE. It is used to let you check the result is correct
        np.random.seed(2019)
        m = 10
        y_check = np.random.randn(m, 1)
        h_check = np.random.randn(m, 1)
        print('y= {}, \nh= {}'.format(y_check, h_check))
        lin_reg_2 = Linear_Regression_2()
        lin_reg_2.m = m

        self.assertEqual(lin_reg_2.J(h_check, y_check), 0.897146515186598)

class CostFunctionTest(unittest.TestCase):

    def test_value(self):
        # DON'T_CHANGE_THIS_CODE. It is used to let you check the result is correct
        np.random.seed(2020)
        m = 10
        n = 1
        X_check = np.random.randn(m, n)
        y_check = np.random.randn(m, 1)
        b_check = np.random.randn()
        w_check = np.random.randn(1, n)
        params = b_check, w_check
        print('X= {}, \ny= {}, \nb= {} \nw= {}'.format(X_check, y_check, b_check, w_check))

        lin_reg_3 = Linear_Regression_3()
        lin_reg_3.m = m
        lin_reg_3.n = n

        self.assertEqual(lin_reg_3.J_derivative(params, X_check, y_check), (round(2.1904608819958713,5), round(-1.4328426209410612,5)) )

class GradientDescent_test(unittest.TestCase):

    def test_value(self):
            # DON'T_CHANGE_THIS_CODE. It is used to let you check the result is correct
            np.random.seed(2021)
            m = 10
            n = 1
            X_check = np.random.randn(m, n)
            y_check = np.random.randn(m, 1)
            print('X= {}, \ny= {}'.format(X_check, y_check))
            lin_reg_4 = Linear_Regression_4(alpha=1, max_iter=5, verbose=1)
            lin_reg_4.fit(X_check, y_check)
            self.assertEqual( lin_reg_4.fit(X_check, y_check), True )