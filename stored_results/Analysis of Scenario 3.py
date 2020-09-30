
# coding: utf-8

# # Analysis of the experiments with different schedulers for Federated Learning - Scenario 3 (achieved makespan, with limits)

# In[ ]:


# modules for the analysis
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")


# - Results with linear costs

# In[ ]:


# reads result file for linear costs
linear_results = pd.read_csv('results_with_linear_costs_and_limits.csv', comment='#')
print('- Results with Linear costs')
linear_results.head(6)


# In[ ]:


# checking the number of results versus the expected number of results
expected_number = 91*6*2  # 91 numbers of tasks, 6 schedulers, 2 numbers of resources
print(f'-- Number of results with linear costs: {len(linear_results)} (expected: {expected_number})')


# In[ ]:


# renaming schedulers
linear_results['Scheduler'] = linear_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)', 'FedAvg', 'Fed-LBAP'],
    ['Ext-Proportional(1)', 'Ext-Proportional(T/n)', 'Ext-Proportional(T)',
     'Ext-FedAvg', 'Ext-Fed-LBAP'])

order = ['Ext-Proportional(1)', 'Ext-Proportional(T/n)', 'Ext-Proportional(T)',
         'Ext-FedAvg', 'Ext-Fed-LBAP', 'OLAR']

# setting colors
c_prop = sns.color_palette("YlOrBr", 10)[5:8]
c_fedavg = [sns.color_palette("colorblind", 10)[4]]
c_fed_lbap = [sns.color_palette("PuOr", 10)[8]]
c_olar = [sns.color_palette("BuGn", 10)[8]]

sns.set_palette(c_prop + c_fedavg + c_fed_lbap + c_olar)


# In[ ]:


print('-- Generating figures')

# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Makespan (a.u.)', fontsize=13)
plt.xticks(range(1000,10001,1000))
plt.xticks(rotation=15)
plt.ylim(0, 12000)

sns.lineplot(data=linear_results[linear_results.Resources == 10],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s3-linear-10.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Makespan (a.u.)', fontsize=13)
plt.xticks(range(1000,10001,1000))
plt.xticks(rotation=15)
plt.ylim(0, 1200)

sns.lineplot(data=linear_results[linear_results.Resources == 100],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s3-linear-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = linear_results[linear_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = linear_results[linear_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or less than OLAR:' +
          f'{greater}, {equal}, {less}.')


# ---
# - Results with quadratic costs

# In[ ]:


# reads result file for quadratic costs
quadratic_results = pd.read_csv('results_with_quadratic_costs_and_limits.csv', comment='#')
print('\n- Results with Quadratic costs')
quadratic_results.head(6)


# In[ ]:


# checking the number of results versus the expected number of results
print(f'-- Number of results with quadratic costs: {len(quadratic_results)} (expected: {expected_number})')


# In[ ]:


# renaming schedulers
quadratic_results['Scheduler'] = quadratic_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)', 'FedAvg', 'Fed-LBAP'],
    ['Ext-Proportional(1)', 'Ext-Proportional(T/n)', 'Ext-Proportional(T)',
     'Ext-FedAvg', 'Ext-Fed-LBAP'])


# In[ ]:


print('-- Generating figures')

# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Makespan (a.u.)', fontsize=13)
plt.xticks(range(1000,10001,1000))
plt.xticks(rotation=15)
plt.ylim(0, 12000000)
plt.ticklabel_format(axis='y', style='plain')

sns.lineplot(data=quadratic_results[quadratic_results.Resources == 10],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s3-quadratic-10.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Makespan (a.u.)', fontsize=13)
plt.xticks(range(1000,10001,1000))
plt.xticks(rotation=15)
plt.ylim(0, 160000)

sns.lineplot(data=quadratic_results[quadratic_results.Resources == 100],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order,
             legend=False)

plt.savefig("s3-quadratic-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = quadratic_results[quadratic_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = quadratic_results[quadratic_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or less than OLAR:' +
          f'{greater}, {equal}, {less}.')

