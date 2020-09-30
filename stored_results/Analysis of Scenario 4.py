
# coding: utf-8

# # Analysis of the experiments with different schedulers for Federated Learning - Scenario 4 (scheduling time, with limits)

# In[ ]:


# modules for the analysis
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sns.set_theme(style="whitegrid")


# - Analysis for a fixed number of resources

# In[ ]:


print('- Analysis for a fixed number of resources')
# reads result file containing performance results for a fixed number of resources
time_fixed_resources = pd.read_csv('results_of_timing_with_fixed_resources_and_limits.csv', comment='#')
time_fixed_resources['ms'] = time_fixed_resources['Time'] * 10
time_fixed_resources.head(5)


# In[ ]:


# checking the number of results versus the expected number of results
expected_number = 10*4*50  # 10 number of tasks, 4 schedulers, 50 samples
print(f'-- Number of results with fixed resources: {len(time_fixed_resources)} (expected: {expected_number})')


# In[ ]:


# rename schedulers to emphasize that some were extended
time_fixed_resources['Scheduler'] = time_fixed_resources['Scheduler'].replace(
    ['Proportional', 'Fed-LBAP', 'FedAvg'],
    ['Ext-Proportional', 'Ext-Fed-LBAP', 'Ext-FedAvg'])


# In[ ]:


# reordering schedulers and setting colors
order = ['Ext-Fed-LBAP', 'OLAR', 'Ext-Proportional', 'Ext-FedAvg']
c_prop = [sns.color_palette("YlOrBr", 10)[7]]
c_fedavg = [sns.color_palette("colorblind", 10)[4]]
c_fed_lbap = [sns.color_palette("PuOr", 10)[8]]
c_olar = [sns.color_palette("BuGn", 10)[8]]

sns.set_palette(c_fed_lbap + c_olar + c_prop + c_fedavg)


# In[ ]:


print('-- Plotting performance graphs')

# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Average execution time (ms)', fontsize=13)
plt.xticks(range(1000,10001,1000))
plt.xticks(rotation=15)

sns.lineplot(data=time_fixed_resources,
             x='Tasks',
             y='ms',
             hue='Scheduler',
             linewidth=2,
             ci='sd',
             hue_order=order)

plt.ylim(0, 70)
plt.xlim(890, 10110)
plt.legend(ncol=1, title='Scheduler', loc='upper left')

plt.savefig("s4-fixed-resources.pdf", bbox_inches='tight')


# In[ ]:


# Averages for different numbers of tasks and schedulers
averages = time_fixed_resources['Time'].groupby([time_fixed_resources['Scheduler'], time_fixed_resources['Tasks']]).mean().unstack(level=0)
averages*10 # 1000 to go from seconds to ms, /100 to consider the cost per call


# In[ ]:


# speedups
print('-- Computing speedups of OLAR over Fed-LBAP')
print(averages['Ext-Fed-LBAP'] / averages['OLAR'])


# Checking the distribution of origin of the different results with 5% confidence.
# The interest is to check if they come from normal distributions.

# In[ ]:


print('-- Checking the distribution of origin for different results')
# Checking all schedulers
np.random.seed(2020)
for scheduler in order:
    for tasks in range(1000,10001,1000):
        results = list(time_fixed_resources[(time_fixed_resources['Scheduler'] == scheduler) &
                                            (time_fixed_resources['Tasks'] == tasks)].Time)
        print(f'Results for scheduler {scheduler} and {tasks} tasks')
        print(stats.kstest(results, 'norm', args=(np.mean(results), np.std(results))))


# We can reject the null hypothesis that some of the results come from a normal distribution with 5% confidence (i.e., some results had a p-value under 0.05).
# In this case, we use Mann-Whitney U test to compare OLAR and Fed-LBAP.

# In[ ]:


print('-- Statistical comparison between OLAR and Fed-LBAP')
olar = list(time_fixed_resources[time_fixed_resources['Scheduler'] == 'OLAR'].Time)
fed_lbap = list(time_fixed_resources[time_fixed_resources['Scheduler'] == 'Ext-Fed-LBAP'].Time)
print('Mann-Whitney U test for comparison between OLAR and Ext-Fed-LBAP (all samples).')
print(stats.mannwhitneyu(olar, fed_lbap, alternative='two-sided'))

for tasks in range(1000,10001,1000):
    olar = list(time_fixed_resources[(time_fixed_resources['Scheduler'] == 'OLAR') &
                                     (time_fixed_resources['Tasks'] == tasks)].Time)
    fed_lbap = list(time_fixed_resources[(time_fixed_resources['Scheduler'] == 'Ext-Fed-LBAP') &
                                         (time_fixed_resources['Tasks'] == tasks)].Time)
    print(f'Mann-Whitney U test for comparison between OLAR and Ext-Fed-LBAP ({tasks} tasks).')
    print(stats.mannwhitneyu(olar, fed_lbap, alternative='two-sided'))


# For all cases, we can say that OLAR and Fed-LBAP perform differently (we reject the null hypothesis that they come from the same distribution, as as p-values < 0.05.

# ---
# - Analysis for a fixed number of tasks

# In[ ]:


print('\n- Analysis for a fixed number of tasks')
# reads result file containing performance results for a fixed number of resources
time_fixed_tasks = pd.read_csv('results_of_timing_with_fixed_tasks_and_limits.csv', comment='#')
time_fixed_tasks['ms'] = time_fixed_tasks['Time'] * 10
time_fixed_tasks.head(5)


# In[ ]:


# checking the number of results versus the expected number of results
print(f'-- Number of results with fixed tasks: {len(time_fixed_tasks)} (expected: {expected_number})')


# In[ ]:


# rename schedulers to emphasize that some were extended
time_fixed_tasks['Scheduler'] = time_fixed_tasks['Scheduler'].replace(
    ['Proportional', 'Fed-LBAP', 'FedAvg'],
    ['Ext-Proportional', 'Ext-Fed-LBAP', 'Ext-FedAvg'])


# In[ ]:


print('-- Plotting performance graphs')
# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of resources (n)', fontsize=13)
plt.ylabel('Average execution time (ms)', fontsize=13)
plt.xticks(range(100,1001,100))
plt.xticks(rotation=15)

sns.lineplot(data=time_fixed_tasks,
             x='Resources',
             y='ms',
             hue='Scheduler',
             linewidth=2,
             ci='sd',
             hue_order=order,
             legend=False)

plt.ylim(0, 70)
plt.xlim(90, 1010)

plt.savefig("s4-fixed-tasks.pdf", bbox_inches='tight')


# In[ ]:


# Averages for different numbers of resources and schedulers
averages = time_fixed_tasks['Time'].groupby([time_fixed_tasks['Scheduler'], time_fixed_tasks['Resources']]).mean().unstack(level=0)
averages*10 # 1000 to go from seconds to ms, /100 to consider the cost per call


# Checking the distribution of origin of the different results with 5% confidence.
# The interest is to check if they come from normal distributions.

# In[ ]:


# speedups
print('-- Computing speedups of OLAR over Fed-LBAP')
print(averages['Ext-Fed-LBAP'] / averages['OLAR'])


# In[ ]:


print('-- Checking the distribution of origin for different results')
# Checking all schedulers
np.random.seed(2020)
for scheduler in order:
    for resources in range(100,1001,100):
        results = list(time_fixed_tasks[(time_fixed_tasks['Scheduler'] == scheduler) &
                                        (time_fixed_tasks['Resources'] == resources)].Time)
        print(f'Results for scheduler {scheduler} and {resources} resources')
        print(stats.kstest(results, 'norm', args=(np.mean(results), np.std(results))))


# We can reject the null hypothesis that some of the results come from a normal distribution with 5% confidence (i.e., some results had a p-value under 0.05).
# In this case, we use Mann-Whitney U test to compare OLAR and Fed-LBAP.

# In[ ]:


print('-- Statistical comparison between OLAR and Fed-LBAP')
olar = list(time_fixed_tasks[time_fixed_tasks['Scheduler'] == 'OLAR'].Time)
fed_lbap = list(time_fixed_tasks[time_fixed_tasks['Scheduler'] == 'Ext-Fed-LBAP'].Time)
print('Mann-Whitney U test for comparison between OLAR and Ext-Fed-LBAP (all samples).')
print(stats.mannwhitneyu(olar, fed_lbap, alternative='two-sided'))

for resources in range(100,1001,100):
    olar = list(time_fixed_tasks[(time_fixed_tasks['Scheduler'] == 'OLAR') &
                                 (time_fixed_tasks['Resources'] == resources)].Time)
    fed_lbap = list(time_fixed_tasks[(time_fixed_tasks['Scheduler'] == 'Ext-Fed-LBAP') &
                                     (time_fixed_tasks['Resources'] == resources)].Time)
    print(f'Mann-Whitney U test for comparison between OLAR and Ext-Fed-LBAP ({resources} resources).')
    print(stats.mannwhitneyu(olar, fed_lbap, alternative='two-sided'))


# For all cases, we can say that OLAR and Fed-LBAP perform differently (we reject the null hypothesis that they come from the same distribution, as as p-values < 0.05.
