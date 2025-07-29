# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:47:55 2025

@author: Sourav
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle

with open("Store_robust_cf_cost","rb") as f:
    cf_list = pickle.load(f)
f.close()

with open("Store_robust_vf_cost","rb") as f:
    vf_list = pickle.load(f)
f.close()
cf_list,vf_list = np.array(cf_list),np.array(vf_list)

lambda_ = 0.1
b = 38
y = np.max([vf_list/lambda_,cf_list-b],axis=0)
'''plt.figure()
#plt.plot(vf_list[0:100])
plt.plot(np.max([vf_list/lambda_,cf_list-b],axis=0)[0:100])
plt.plot(cf_list[0:100])
plt.plot(np.ones(100)*b)
plt.show()'''

with open("Epi_RC_objective_MR_4s_2a","rb") as f:
    evf_list = pickle.load(f)
f.close()
#evf_list.insert(0,0)

with open("Epi_RC_constrainte_MR_4s_2a","rb") as f:
    ecf_list = pickle.load(f)
f.close()
#ecf_list.insert(0,0)
ey= np.maximum(np.array(evf_list),(b - np.array(ecf_list)))
#y= np.maximum(np.array(vf_list),(b - np.array(cf_list)))
x= np.arange(len(vf_list))
plt.figure()
plt.plot(x,y,alpha=0.6)
plt.plot(x,ey,alpha=0.6)
plt.xlabel('Iteration')
plt.ylabel('max(Vf/lambda,Vf-b)')
plt.title("Maximum violation(MR setting nS=4,nA=2)")
plt.savefig('Maximum_violation_1000_steps.pdf')
plt.legend(['Our_algo','Epi_RC'])
plt.show()


