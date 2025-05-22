# -*- coding: utf-8 -*-
"""
Created on Mon May 19 16:13:46 2025

@author: gangu
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

lambda_ = 50
baseline = 60

with open("Value_function_kl_lambda_cleaning_16_4_50_upload.pkl","rb") as f:
    vf = pickle.load(f)
f.close()
with open("Cost_function_kl_lambda_cleaning_16_4_50_upload.pkl","rb") as f:
    cf = pickle.load(f)
f.close()

with open("Store_robust_vf_cost_GC","rb") as f:
    rvf = pickle.load(f)
f.close()
with open("Store_robust_cf_cost_GC","rb") as f:
    rcf = pickle.load(f)
f.close()

with open("Epi_RC_objective_Garbage_collector_new_set_upload","rb") as f:
    evf = pickle.load(f)
f.close()
with open("Epi_RC_constrainte_Garbage_collector_new_set_upload","rb") as f:
    ecf = pickle.load(f)
f.close()


x = np.arange(len(vf)+1)
plt.figure()
plt.plot(vf,color='green')
plt.plot(rvf)
plt.plot(evf[0:1000])
plt.ylabel("Cumulative Objective reward",fontweight=200,fontsize=18)
plt.xlabel("Iteration",fontweight=200,fontsize=18)
plt.title('Objective function (Garbage_collector)',fontweight=200,fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['RNPG','RPPG','EPIRC_PGS'])
plt.savefig('Vf_update_GC.pdf')
plt.show()

plt.figure()
plt.plot(cf,color='green')
plt.plot(rcf)
plt.plot(ecf[0:1000])
plt.plot(np.ones(len(vf))*baseline,linestyle='-.',color='black',linewidth=1)
plt.ylabel("Cumulative constraint cost",fontweight=200,fontsize=18)
plt.xlabel("Iteration",fontweight=200,fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.fill_between(x, 0,baseline, color='blue', alpha=0.1, label='Safe region')
# Shade the region between y1 and y2 where y1 < y2
plt.fill_between(x,baseline, 300, color='red', alpha=0.1, label='UnSafe_region')
plt.legend(['RNPG','RPPG','EPIRC_PGS','baseline','Safe region','Unsafe Region'],fontsize=18)
plt.title('Constraint function (Garbage collector)',fontweight=200,fontsize=18)
plt.savefig('CF_update_GC.pdf')
plt.show()