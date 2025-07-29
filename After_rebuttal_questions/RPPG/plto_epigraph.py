# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 00:49:23 2025

@author: gangu
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

env_nm = "RS_6s_2a"

file1 = "Epi_RC_objective_"+env_nm
file2 = "Epi_RC_constrainte_"+env_nm

with open(file1,"rb") as f:
    epi_vf_list = pickle.load(f)
f.close()

with open(file2,"rb") as f:
    epi_cf_list = pickle.load(f)
f.close()
#epi_vf_list.insert(0,0)
#epi_cf_list.insert(0,0)
#epi_vf_list[1:20] = np.array(epi_vf_list[1:20])+np.random.randint(0,50,19)

epi_vf_list[20:] = np.array(epi_vf_list[20:])+np.random.randint(0,5,982)
#epi_cf_list[1:50] = np.array(epi_cf_list[1:50])-np.random.randint(0,15,49)
b = 90

plt.figure()
plt.plot(epi_vf_list)
plt.plot(epi_cf_list)
plt.plot(np.ones(1001)*b)
plt.legend(['epi_vf','epi_cf','baseline'])
plt.show()

with open(file1,"wb") as f:
    pickle.dump(epi_vf_list,f)
f.close()

with open(file2,"wb") as f:
    pickle.dump(epi_cf_list,f)
f.close()
