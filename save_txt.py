# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:41:11 2021

@author: 66441
"""

import numpy as np

a = np.arange(0,10).reshape(2,-1)
print("a:\n", a)
np.save("a.npy", a)
b = np.load("a.npy")
print("b:\n", b)