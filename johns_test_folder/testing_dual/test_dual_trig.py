import unittest
import dualpy as dp 
import numpy as np


class TestDualTrig(unittest.TestCase):

    def runTest(self):
        tv = np.linspace(0,60,61)
        t = dp.seed(tv, "tv")
        self.assertEqual(t.jacobians,t.jacobians,"not a jacobian ! ! ! ")


unittest.main()