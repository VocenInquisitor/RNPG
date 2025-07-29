# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:45:10 2025

@author: gangu
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_excel('tvf_and_tcf_data_with_uncertainity_new_way_eps_1.xlsx')
#vf = data['vf'][::150]
#cf = data['cf'][::150]
vf = data['reward']
cf =data['cost']
y= cf.values
x = vf.values
print(len(vf))


plt.plot(np.arange(1,len(vf)+1),vf)
plt.plot(np.cumsum(x)/np.arange(1,len(vf)+1))
plt.xlabel('Episode number')
plt.ylabel('Total value function reached')
plt.title('Episode wise Value function')
plt.savefig('Cartpole_episode_wise_vf_CMDP setting.pdf')
#plt.plot(np.ones(1000)*200)
plt.legend(['vf value','average_vf'])
plt.savefig('RCMDP_setting_Cartpole_vf.pdf')
plt.show()

plt.figure()
plt.plot(y,label='episode wise cost function')
plt.axhspan(0, 200, color='blue', alpha=0.2, label='Safe region')

# Unsafe region: y > 100
plt.axhspan(200, max(y.max(), 500), color='red', alpha=0.2, label='Unsafe region')

# Optional safety limit line
plt.axhline(200, color='red', linestyle='--', linewidth=1)

# Decorate
plt.xlabel('Time step / Episode')
plt.ylabel('Expected Cost')
plt.title('Expected Cost vs. Safety Limit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('RCMDP_setting_cartpole_cost.pdf')
plt.show()

