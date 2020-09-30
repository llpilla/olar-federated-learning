
# coding: utf-8

# # Analysis of the experiments with different schedulers for Federated Learning - Scenario 1 (achieved makespan, no limits)

# In[ ]:


# modules for the analysis
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")


# - Results with recursive costs

# In[ ]:


# reads result file for recursive costs
recursive_results = pd.read_csv('results_with_recursive_costs.csv', comment='#')
print('- Results with Recursive costs')
recursive_results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
expected_number = 91*9*2  # 91 numbers of tasks, 9 schedulers, 2 numbers of resources
print(f'-- Number of results with recursive costs: {len(recursive_results)} (expected: {expected_number})')


# In[ ]:


# renaming and reordering schedulers
recursive_results['Scheduler'] = recursive_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)',
     'Random-(seed:1000)',  'Random-(seed:2000)', 'Random-(seed:3000)'],
    ['Proportional(1)', 'Proportional(T/n)', 'Proportional(T)',
     'Random(\u2660)',  'Random(\u2663)', 'Random(\u2665)'])

order = ['Random(\u2660)',  'Random(\u2663)', 'Random(\u2665)',
         'Proportional(1)', 'Proportional(T/n)', 'Proportional(T)',
         'FedAvg', 'Fed-LBAP', 'OLAR']


# In[ ]:


# setting colors
c_random = sns.color_palette("GnBu", 10)[5:8]
c_prop = sns.color_palette("YlOrBr", 10)[5:8]
c_fedavg = [sns.color_palette("colorblind", 10)[4]]
c_fed_lbap = [sns.color_palette("PuOr", 10)[8]]
c_olar = [sns.color_palette("BuGn", 10)[8]]

sns.set_palette(c_random + c_prop + c_fedavg + c_fed_lbap + c_olar)


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

sns.lineplot(data=recursive_results[recursive_results.Resources == 10],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s1-recursive-10.pdf", bbox_inches='tight')


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
plt.ylim(0, 1400)

sns.lineplot(data=recursive_results[recursive_results.Resources == 100],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order,
             legend=False)

plt.savefig("s1-recursive-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = recursive_results[recursive_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = recursive_results[recursive_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or smaller than OLAR: ' +
          f'{greater}, {equal}, {less}.')


# ---
# - Results with linear costs

# In[ ]:


# reads result file for linear costs
linear_results = pd.read_csv('results_with_linear_costs.csv', comment='#')
print('\n- Results with Linear costs')
linear_results.head(9)


# In[ ]:


print(f'-- Number of results with linear costs: {len(linear_results)} (expected: {expected_number})')


# In[ ]:


# renaming schedulers
linear_results['Scheduler'] = linear_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)',
     'Random-(seed:4000)',  'Random-(seed:5000)', 'Random-(seed:6000)'],
    ['Proportional(1)', 'Proportional(T/n)', 'Proportional(T)',
     'Random(\u2660)',  'Random(\u2663)', 'Random(\u2665)'])


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

plt.savefig("s1-linear-10.pdf", bbox_inches='tight')


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
             hue_order=order,
             legend=False)

plt.savefig("s1-linear-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = linear_results[linear_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = linear_results[linear_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or smaller than OLAR: ' +
          f'{greater}, {equal}, {less}.')


# ---
# - Results with nlogn costs

# In[ ]:


# reads result file for nlogn costs
nlogn_results = pd.read_csv('results_with_nlogn_costs.csv', comment='#')
print('\n- Results with Nlogn costs')
nlogn_results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
print(f'-- Number of results with nlogn costs: {len(nlogn_results)} (expected: {expected_number})')


# In[ ]:


# renaming and reordering schedulers
nlogn_results['Scheduler'] = nlogn_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)',
     'Random-(seed:7000)',  'Random-(seed:8000)', 'Random-(seed:9000)'],
    ['Proportional(1)', 'Proportional(T/n)', 'Proportional(T)',
     'Random(\u2660)',  'Random(\u2663)', 'Random(\u2665)'])


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
plt.ylim(0, 80000)

sns.lineplot(data=nlogn_results[nlogn_results.Resources == 10],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s1-nlogn-10.pdf", bbox_inches='tight')


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
plt.ylim(0, 6000)

sns.lineplot(data=nlogn_results[nlogn_results.Resources == 100],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order,
             legend=False)

plt.savefig("s1-nlogn-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = nlogn_results[nlogn_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = nlogn_results[nlogn_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or smaller than OLAR: ' +
          f'{greater}, {equal}, {less}.')


# ---
# - Results with quadratic costs

# In[ ]:


# reads result file for quadratic costs
quadratic_results = pd.read_csv('results_with_quadratic_costs.csv', comment='#')
print('\n- Results with Quadratic costs')
quadratic_results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
print(f'-- Number of results with quadratic costs: {len(quadratic_results)} (expected: {expected_number})')


# In[ ]:


# renaming schedulers
quadratic_results['Scheduler'] = quadratic_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)',
     'Random-(seed:10000)',  'Random-(seed:11000)', 'Random-(seed:12000)'],
    ['Proportional(1)', 'Proportional(T/n)', 'Proportional(T)',
     'Random(\u2660)',  'Random(\u2663)', 'Random(\u2665)'])


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
plt.ylim(0, 16000000)
plt.ticklabel_format(axis='y', style='plain')

sns.lineplot(data=quadratic_results[quadratic_results.Resources == 10],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s1-quadratic-10.pdf", bbox_inches='tight')


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
plt.ylim(0, 140000)

sns.lineplot(data=quadratic_results[quadratic_results.Resources == 100],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order,
             legend=False)

plt.savefig("s1-quadratic-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = quadratic_results[quadratic_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = quadratic_results[quadratic_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or smaller than OLAR: ' +
          f'{greater}, {equal}, {less}.')


# ---
# - Results with mixed costs

# In[ ]:


# reads result file for random costs
mixed_results = pd.read_csv('results_with_mixed_costs.csv', comment='#')
print('\n- Results with Mixed costs')
mixed_results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
print(f'-- Number of results with mixed costs: {len(mixed_results)} (expected: {expected_number})')


# In[ ]:


# renaming schedulers
mixed_results['Scheduler'] = mixed_results['Scheduler'].replace(
    ['Proportional-(1)', 'Proportional-(tasks/res)', 'Proportional-(tasks)',
     'Random-(seed:13000)',  'Random-(seed:14000)', 'Random-(seed:15000)'],
    ['Proportional(1)', 'Proportional(T/n)', 'Proportional(T)',
     'Random(\u2660)',  'Random(\u2663)', 'Random(\u2665)'])


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
plt.ylim(0, 8000000)
plt.ticklabel_format(axis='y', style='plain')

sns.lineplot(data=mixed_results[mixed_results.Resources == 10],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order)

plt.savefig("s1-mixed-10.pdf", bbox_inches='tight')


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
plt.ylim(0, 120000)

sns.lineplot(data=mixed_results[mixed_results.Resources == 100],
             x='Tasks',
             y='Makespan',
             hue='Scheduler',
             linewidth=2,
             hue_order=order,
             legend=False)

plt.savefig("s1-mixed-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of OLAR
olar_makespan = mixed_results[mixed_results['Scheduler'] == 'OLAR'].Makespan.reset_index(drop=True)
for scheduler in order[:-1]:
    other_makespan = mixed_results[mixed_results['Scheduler'] == scheduler].Makespan.reset_index(drop=True)
    greater = np.sum(other_makespan > olar_makespan)
    equal = np.sum(other_makespan == olar_makespan)
    less = np.sum(other_makespan < olar_makespan)
    print(f'Number of times {scheduler} provides a makespan greater, equal, or smaller than OLAR: ' +
          f'{greater}, {equal}, {less}.')


# In[ ]:


# Comparing the optimal assignment to FedAvg with 10000 tasks
optimal_result = mixed_results.loc[(mixed_results['Scheduler'] == 'OLAR') &
                                   (mixed_results['Tasks'] == 10000) &
                                   (mixed_results['Resources'] == 10)]
fedavg_result = mixed_results.loc[(mixed_results['Scheduler'] == 'FedAvg') &
                                   (mixed_results['Tasks'] == 10000) &
                                   (mixed_results['Resources'] == 10)]
print(f'Makespan for the optimal assignment for 10000 tasks and 10 resources: {float(optimal_result.Makespan)}')
print(f'Makespan for FedAVG\'s assignment for 10000 tasks and 10 resources : {float(fedavg_result.Makespan)}')
print(f'Ratio: {float(fedavg_result.Makespan)/float(optimal_result.Makespan)}')

