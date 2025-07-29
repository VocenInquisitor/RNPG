# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:43:57 2025

@author: gangu
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_excel('New_cartpole_tvf_and_tcf_store.xlsx')
vf = data['vf']
cf = data['cf']
#print(len(cf))
time = cf.index
x = cf

# Step 2: Draw safe/unsafe regions as vertical spans
plt.axvspan(0, 3, color='blue', alpha=0.3, label='Safe region')
plt.axvspan(3, 5, color='red', alpha=0.3, label='Unsafe region')
#plt.axvspan(2, 4.8, color='red', alpha=0.3)

# Step 3: Plot cart x-positions vertically (x = cart position, y = time)
plt.plot(x, time, marker='o', color='black', label='Cart Position')

# Step 4: Labels and legend
plt.xlabel('Cart Position (x)')
plt.ylabel('Time Step')
plt.title('CartPole Trajectory with Safe/Unsafe Regions')
plt.xlim(0, 5)
plt.grid(True)
plt.legend()
plt.savefig('Cartpole_positions_plot.pdf')
plt.gca().invert_yaxis()  # Optional: time goes downward

plt.show()

#Example: assume cf is already defined as a DataFrame with one column
#cf = pd.DataFrame({'x': your_cart_positions})

#Step 1: Classify each position as 'Safe' or 'Unsafe'
data['region'] = data['cf'].apply(lambda x: 'Safe' if x < 3 else 'Unsafe')

# Step 2: Plot the countplot
sns.countplot(data=data, x='region', palette={'Safe': 'blue', 'Unsafe': 'red'})

# Step 3: Decorate
plt.title('Number of Timesteps in Safe vs Unsafe Region')
plt.ylabel('Count')
plt.xlabel('Region')
plt.grid(axis='y')
plt.savefig('Frequency_of_safe_and_unsafe_region_landing.pdf')
plt.show()
plt.plot(vf)
plt.xlabel('Batch number')
plt.title('Batch wise value function')
plt.ylabel('Value function')
plt.savefig('Value function plot.pdf')
plt.show()

plt.figure()
plt.plot(np.cumsum(vf)/np.arange(1,len(vf)+1))
plt.xlabel('Batch number')
plt.title('Batch wise value function')
plt.ylabel('Average long run reward')
plt.savefig('Running average value function plot.pdf')
plt.show()