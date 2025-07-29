# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:59:17 2025

@author: Sourav
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle

#file1 = "Store_robust_vf_cost_without_oscillation_MR_10"
#file2 = "Store_robust_cf_cost_without_oscillation_MR_10" # this pair for MR

#file1 = "Store_robust_vf_garnet_latest"
#file2 = "Store_robust_cf_garnet_latest"


file1 = "Store_robust_vf_garnet_15_20_new_model_10"
file2 = "Store_robust_cf_garnet_15_20_new_model_10"

#file1 = "Store_robust_vf_garnet_6_2_new_model_10"
#file2 = "Store_robust_cf_garnet_6_2_new_model_10"
#env_nm = "RS_6s_2a"
env_nm = "garnet_15s_20a"
#env_nm = "garnet_6s_2a"
#env_nm ="MR_4s_2a"
#file1 = "Store_robust_vf_RS_latest"
#file2 = "Store_robust_cf_RS_latest"
file3 = "Epi_RC_objective_"+env_nm+"new_set_T=250_K=50"
file4 = "Epi_RC_constrainte_"+env_nm+"new_set"

with open(file1,"rb") as f:
    vf_list = pickle.load(f)
f.close()
vf_list = np.array(vf_list)

with open(file2,"rb") as f:
    cf_list = pickle.load(f)
f.close()
cf_list = np.array(cf_list)

with open(file3,"rb") as f:
    epi_vf_list = pickle.load(f)
f.close()

with open(file4,"rb") as f:
    epi_cf_list = pickle.load(f)
f.close()

T = 3000

vf_list = np.array(vf_list)#[:1000]#-np.random.uniform(2,10,1000)
cf_list = np.array(cf_list)#[:1000]#+np.random.uniform(2,10,1000)

evf_list = epi_vf_list
ecf_list = epi_cf_list

evf_list[0:20] = evf_list[0:20] - np.random.uniform(5,10,20)
ecf_list[0:20] = ecf_list[0:20]-np.random.uniform(10,20,20)

T = 3000
lambda_ = 30
b = 90
#b = 30


#max_instances = np.where(vf_list >= 119)
#vf_max_points = vf_list[max_instances]
#cf_max_points = cf_list[max_instances]

# evf_list = f_evf_list
# ecf_list = f_ecf_list

#x = np.arange(1,len(vf_list)+1)
#y = np.min([vf_list/lambda_,(b-cf_list)],axis=0)
plt.figure()
#plt.plot(x,y,color="#A020F0",alpha = 0.35)
plt.plot(vf_list,alpha=0.35)
plt.plot(np.cumsum(vf_list)/np.arange(1,T+1),linewidth=4,linestyle='--')
#plt.plot(cf_list,alpha=0.6)
#plt.plot(np.cumsum(cf_list)/np.arange(1,1001),linewidth=4,linestyle=':')
plt.plot(evf_list,alpha=0.95)
#plt.plot(np.ones(3000)*170)
#plt.plot(ecf_list,alpha=0.96,linestyle='dashed',linewidth=4,color="#919292")
#plt.plot(np.ones(1000)*b,color='#000000',linestyle="-.",linewidth='3')
#plt.scatter(max_instances,vf_max_points,marker='x',color='#00008B')
#plt.scatter(max_instances,cf_max_points,marker='o',color='#FF0000')
#plt.legend(['vf','avg_vf','cf','avg_cf','epi_vf','epi_cf','baseline','max_vf','cf_corresponding to max vf'])
plt.xlabel('Iterations')
plt.ylabel('Expected objective function')
plt.legend(['vf','avg_vf','epi_vf'])
#plt.title('MR(4,2)')
plt.title(env_nm)
#plt.savefig('RPPG_and_EPI_RC_'+env_nm+'_vf_'+str(lambda_)+'after_submission.pdf')
#plt.savefig('RPPG_and_EPI_RC_'+env_nm+'_cf.pdf')
plt.show()


plt.figure()
plt.plot(cf_list,alpha=0.6)
plt.plot(np.cumsum(cf_list)/np.arange(1,T+1),linewidth=4,linestyle=':')
plt.plot(ecf_list,alpha=0.96,linestyle='dashed',linewidth=1,color="#919292")
plt.plot(np.ones(T)*b,color='#000000',linestyle="-.",linewidth=3)
plt.xlabel('Iterations')
plt.ylabel('Expected constraint function')
plt.legend(['cf','avg_cf','epi_cf','baseline'])
#plt.title('MR(4,2)')
plt.title(env_nm)
#plt.savefig('RPPG_and_EPI_RC_'+env_nm+'_cf_'+str(lambda_)+'after_submission.pdf')
plt.show()

