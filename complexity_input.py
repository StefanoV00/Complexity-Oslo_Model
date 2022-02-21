# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:27:23 2022

@author: Stefano
"""
import numpy as np

#FOR COMPLEXITY TESTING
######################################################################
T_A = int(1e4)     # for report results int(10e6)
tests_B = [16, 32] # for report results [16, 32, 64, 128]
tests_C = [4, 5]   # for report results [4,5,6]
T_C     = int(1e3) # for report results int(10e3)
Ns_C    = 500      # for report results 2000


#FOR MAIN PROJECT
######################################################################
Llist = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
# for report np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])

M_in   = 10 # for report 30
n      = 1  # for report 3
Tlow_M = int(0.1e6) # for report 5e6

T      = int(0.5e6) # for report int(20e6)
Tlow   = int(5e4)   # for report int(5e6)
Limin_in= 2         # for report 4


#FOR BULKH
######################################################################
Llist_bulkh = np.array([64, 128, 256])
# for report np.array([64, 128, 256, 512])

T_bulkh    = int(0.1e6) # for report int(20e6)
Tlow_bulkh = int(5e4)   # for report int(5e6)


#FOR BULKZ
######################################################################
Llist_bulkz = np.array([64, 128, 256])
# for report np.array([64, 128, 256, 512])
# additional note: this one might take time

T_bulkz    = int(0.1e6) # for report int(20e6)
Tlow_bulkz = int(5e4)   # for report int(5e6)