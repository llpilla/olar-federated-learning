"""# Description of the experiment:
#
# - We generate the costs to up to 10.000 tasks for
# 100 to 1.000 resources with steps of 100.
# - All costs follow linear functions with RNG seeds [0..999].
# - We schedule 10.000 tasks.
# - We run OLAR, Fed-LBAP, FedAvg, Random, and Proportional.
# - We use no lower or upper limits.
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
logger = support.Logger('results_of_timing_with_fixed_tasks.csv')
tasks = 10000
min_resources = 100
max_resources = 1001
step_resources = 100
size_of_sample = 100
number_of_samples = 50
shuffle_initial_seed = 1000
scheduler_name = ['Random', 'Proportional', 'FedAvg', 'Fed-LBAP', 'OLAR']


def run_timing():
    # Stores the description of the experiments
    logger.header(__doc__)
    # Header of the CSV file
    logger.store('Scheduler,Tasks,Resources,Time')
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
    for resources in range(min_resources, max_resources, step_resources):
        print(f'- Running experiments with {resources} resources')
        # sets the string to be used for each scheduler
        calls = []
        # version that verifies the results
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.random(tasks, {resources}, seed_for_random)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.proportional(tasks, {resources}, cost, k)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.fedavg(tasks, {resources})))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.fed_lbap(tasks, {resources}, cost)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.olar(tasks, {resources}, cost, lower_limit, upper_limit)))")
        # version that does not verify the results
        calls.append(f"a = schedulers.random(tasks, {resources}, seed_for_random)")
        calls.append(f"a = schedulers.proportional(tasks, {resources}, cost, k)")
        calls.append(f"a = schedulers.fedavg(tasks, {resources})")
        calls.append(f"a = schedulers.fed_lbap(tasks, {resources}, cost)")
        calls.append(f"a = schedulers.olar(tasks, {resources}, cost, lower_limit, upper_limit)")

        # Setup to generate 100 resources with 10001 costs
        setup = f"""
import numpy as np
import code.schedulers as schedulers
import code.devices as devices
import code.support as support

rng_seed_resources = 0
tasks = 10000
seed_for_random = 1000
resources = {resources}
k = 1
# Initializes the cost matrix with zeros
cost = np.zeros(shape=(resources, tasks+1))
# Fills the cost matrix with costs based on a linear function
base_seed = rng_seed_resources
for i in range(resources):
    devices.create_linear_costs(base_seed, cost, i, tasks)
    base_seed += 1
# Prepares the upper and lower limit arrays
lower_limit = np.zeros(shape=resources, dtype=int)
upper_limit = np.full(shape=resources, fill_value=tasks+1, dtype=int)
"""

        # gathers all samples for a given (number of tasks, scheduler)
        for sample in range(number_of_samples):
            # sets the RNG seed and generates an order of execution
            np.random.seed(shuffle_initial_seed + rounds)
            rounds += 1
            order = np.arange(5)  # five different schedulers
            np.random.shuffle(order)  # random order

            # gathers samples for all schedulers
            for i in order:
                # gathers one sample
                timing = timeit.timeit(setup=setup,
                                       stmt=calls[i],
                                       number=size_of_sample)
                # stores the timing information
                logger.store(f'{scheduler_name[i]},{tasks},{resources},{timing}')


if __name__ == '__main__':
    run_timing()
