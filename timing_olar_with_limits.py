"""# Description of the experiment:
#
# - We generate the costs to up to 10.000 tasks for
# 100 to 1.000 resources with steps of 100.
# - All costs follow linear functions with RNG seeds [0..999].
# - We schedule 10.000 tasks.
# - We run OLAR, Fed-LBAP, FedAvg, Random, and Proportional.
# - All resources have a lower limit of 4, except the resource with the
# highest cost for the number of tasks. That resource gets a minimum of
# (tasks/resources)/4.
# - All resources have an upper limit of 2*(tasks/resources), except the
# resource with the lowest cost for the number of tasks. That resource
# gets an upper limit of (tasks/resources)/2.
# - Each sample is composed of 100 executions of the schedulers.
# - We get 50 samples for each pair (scheduler, tasks)
# - The order of execution of the different schedulers is
# randomly defined. We set an initial RNG seed = 1000 and increase
# it every time we need a new order.
"""

import timeit
import numpy as np
import code.support as support

# File containing the results
logger = support.Logger('results_of_timing_with_fixed_tasks_for_olar_only.csv')

tasks = 10000
size_of_sample = 100
number_of_samples = 50
shuffle_initial_seed = 20000
setup_name = ['Both', 'None', 'Upper', 'Lower']


def run_timing():
    # Stores the description of the experiments
    logger.header(__doc__)
    # Header of the CSV file
    logger.store('Limits,Tasks,Resources,Time')
    # Runs experiments for 10000 tasks
    run_for_fixed_tasks()
    # Finishes logging
    logger.finish()


def run_for_fixed_tasks():
    """
    Runs experiments for a fixed number of tasks.
    """

    # counting the rounds of the experiment to update the RNG seed
    rounds = 0
    # runs experiments for all numbers of resources
    for resources in [100, 500, 1000]:
        print(f'- Running experiments with {resources} resources')
        # sets the string to be used for each scheduler
        # version that verifies the results
        call = f"a = schedulers.olar(tasks, {resources}, cost, lower_limit, upper_limit)"

        # Setup to generate 100 resources with 10001 costs
        avg_tasks = tasks // resources
        setups = []
        setups.append(f"""
import numpy as np
import code.schedulers as schedulers
import code.devices as devices
import code.support as support

rng_seed_resources = 0
tasks = 10000
resources = {resources}
k = 1
# Initializes the cost matrix with zeros
cost = np.zeros(shape=(resources, tasks+1))
# Fills the cost matrix with costs based on a linear function
base_seed = rng_seed_resources
for i in range(resources):
    devices.create_linear_costs(base_seed, cost, i, tasks)
    base_seed += 1
lower_limit = np.full(shape=resources, fill_value=4)
upper_limit = np.full(shape=resources, fill_value={avg_tasks}*2)
# Finds the resource with the maximum and minimum costs for 'tasks'
max_index = np.argmax(cost[:, tasks])
lower_limit[max_index] = {avg_tasks} // 4
min_index = np.argmin(cost[:, tasks])
upper_limit[min_index] = {avg_tasks} // 2
""")
        setups.append(f"""
import numpy as np
import code.schedulers as schedulers
import code.devices as devices
import code.support as support

rng_seed_resources = 0
tasks = 10000
resources = {resources}
k = 1
# Initializes the cost matrix with zeros
cost = np.zeros(shape=(resources, tasks+1))
# Fills the cost matrix with costs based on a linear function
base_seed = rng_seed_resources
for i in range(resources):
    devices.create_linear_costs(base_seed, cost, i, tasks)
    base_seed += 1
lower_limit = np.full(shape=resources, fill_value=0)
upper_limit = np.full(shape=resources, fill_value=tasks+1)
""")

        setups.append(f"""
import numpy as np
import code.schedulers as schedulers
import code.devices as devices
import code.support as support

rng_seed_resources = 0
tasks = 10000
resources = {resources}
k = 1
# Initializes the cost matrix with zeros
cost = np.zeros(shape=(resources, tasks+1))
# Fills the cost matrix with costs based on a linear function
base_seed = rng_seed_resources
for i in range(resources):
    devices.create_linear_costs(base_seed, cost, i, tasks)
    base_seed += 1
lower_limit = np.full(shape=resources, fill_value=0)
upper_limit = np.full(shape=resources, fill_value={avg_tasks}*2)
# Finds the resource with the maximum and minimum costs for 'tasks'
min_index = np.argmin(cost[:, tasks])
upper_limit[min_index] = {avg_tasks} // 2
""")

        setups.append(f"""
import numpy as np
import code.schedulers as schedulers
import code.devices as devices
import code.support as support

rng_seed_resources = 0
tasks = 10000
resources = {resources}
k = 1
# Initializes the cost matrix with zeros
cost = np.zeros(shape=(resources, tasks+1))
# Fills the cost matrix with costs based on a linear function
base_seed = rng_seed_resources
for i in range(resources):
    devices.create_linear_costs(base_seed, cost, i, tasks)
    base_seed += 1
lower_limit = np.full(shape=resources, fill_value=4)
upper_limit = np.full(shape=resources, fill_value=tasks+1)
# Finds the resource with the maximum and minimum costs for 'tasks'
max_index = np.argmax(cost[:, tasks])
lower_limit[max_index] = {avg_tasks} // 4
""")

        # gathers all samples for a given (number of tasks, scheduler)
        for sample in range(number_of_samples):
            # sets the RNG seed and generates an order of execution
            np.random.seed(shuffle_initial_seed + rounds)
            rounds += 1
            order = np.arange(4)  # five different schedulers
            np.random.shuffle(order)  # random order

            # gathers samples for all schedulers
            for i in order:
                # gathers one sample
                timing = timeit.timeit(setup=setups[i],
                                       stmt=call,
                                       number=size_of_sample)
                # stores the timing information
                logger.store(f'{setup_name[i]},10000,{resources},{timing}')


if __name__ == '__main__':
    run_timing()
