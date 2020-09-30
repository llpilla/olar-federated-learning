
# coding: utf-8

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


# - Analysis

# In[ ]:


print('- Analysis for a fixed number of tasks')
# reads result file containing performance results for a fixed number of resources
results = pd.read_csv('results_of_timing_with_fixed_tasks_for_olar_only.csv', comment='#')
results['ms'] = results['Time'] * 10
results.head(4)


# In[ ]:


# Averages for different numbers of tasks and schedulers
averages = results['Time'].groupby([results['Limits'], results['Resources']]).mean().unstack(level=0)
print(averages*10) # 1000 to go from seconds to ms, /100 to consider the cost per call


# Checking the distribution of origin of the different results with 5% confidence.
# The interest is to check if they come from normal distributions.

# In[ ]:


print('-- Checking the distribution of origin for different results')

np.random.seed(2020)
for limit in ['Both', 'Lower', 'None', 'Upper']:
    for resources in [100, 500, 1000]:
        r = list(results[(results['Limits'] == limit) &
                         (results['Resources'] == resources)].Time)
        print(f'Results for limit {limit} and {resources} resources')
        print(stats.kstest(r, 'norm', args=(np.mean(r), np.std(r))))


# We can reject the null hypothesis that some of the results come from a normal distribution with 5% confidence (i.e., some results had a p-value under 0.05).
# In this case, we use Mann-Whitney U test to compare OLAR and Fed-LBAP.

# In[ ]:


limits = ['Both', 'Lower', 'None', 'Upper']
print('-- Statistical comparison between results')

for i in range(len(limits)):
    for j in range(i+1,len(limits)):
        print(f'-- Results for {limits[i]} and {limits[j]}')
        for resources in [100, 500, 1000]:
            res1 = list(results[(results['Limits'] == limits[i]) &
                            (results['Resources'] == resources)].Time)
            res2 = list(results[(results['Limits'] == limits[j]) &
                            (results['Resources'] == resources)].Time)
            st, pv = stats.mannwhitneyu(res1, res2, alternative='two-sided')
            print(f'For {resources} resources, p-value = {pv}')


# All times are considered different among themselves.
