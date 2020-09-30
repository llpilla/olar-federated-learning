"""# Description of the experiment:
#
# - We generate the costs to up to 10.000 tasks for 10 and 100 resources.
# All costs follow nlogn functions with RNG seeds [200..299].
# - We schedule from 1.000 to 10.000 tasks in increments of 100.
# - We run OLAR, Fed-LBAP, FedAvg, Random, and Proportional.
# -- For Random, we generate three different results using three
# starting RNG seeds (7000, 8000, 9000). We increment the starting RNG seed
# each time we change the number of tasks.
# -- For Proportional, we generate three different results using
# three different values for k (1, #tasks/#resources, #tasks)
# - We use no lower or upper limits.
# - Every result is verified and logged to a CSV file.
"""

import numpy as np
import code.support as support
import code.schedulers as schedulers
import code.devices as devices


# File containing the results
logger = support.Logger('results_with_nlogn_costs.csv')
rng_seed_resources = 200
min_tasks = 1000
max_tasks = 10001
step_tasks = 100
seeds_for_random = [7000, 8000, 9000]


def run_nlogn_costs():
    # Stores the description of the experiments
    logger.header(__doc__)
    # Header of the CSV file
    logger.store('Scheduler,Tasks,Resources,Makespan')
    # Runs experiments for 10 resources
    run_for_n_resources(10)
    # Runs experiments for 100 resources
    run_for_n_resources(100)
    # Finishes logging
    logger.finish()


def run_for_n_resources(resources):
    """
    Runs experiments for a number of resources.

    Parameters
    ----------
    resources : int
        Number of resources
    """
    print(f'- Running experiment for {resources} resources.')
    # Initializes the cost matrix with zeros
    cost = np.zeros(shape=(resources, max_tasks+1))
    # Fills the cost matrix with costs based on a nlogn function
    base_seed = rng_seed_resources
    for i in range(resources):
        devices.create_nlogn_costs(base_seed, cost, i, max_tasks)
        base_seed += 1
    # Prepares the upper and lower limit arrays
    lower_limit = np.zeros(shape=resources, dtype=int)
    upper_limit = np.full(shape=resources, fill_value=max_tasks+1, dtype=int)

    # Iterates over the number of tasks running all schedulers
    iteration = 0
    for tasks in range(min_tasks, max_tasks, step_tasks):
        # 1. Run Random with three seeds
        for seed in seeds_for_random:
            a = schedulers.random(tasks, resources, seed + iteration)
            check_and_store(f'Random-(seed:{seed})', tasks, resources, a, cost)
        iteration += 1

        # 2. Run Proportional with three task values to base itself
        a = schedulers.proportional(tasks, resources, cost, 1)
        check_and_store(f'Proportional-(1)', tasks, resources, a, cost)
        a = schedulers.proportional(tasks, resources, cost, tasks//resources)
        check_and_store(f'Proportional-(tasks/res)', tasks, resources, a, cost)
        a = schedulers.proportional(tasks, resources, cost, tasks)
        check_and_store(f'Proportional-(tasks)', tasks, resources, a, cost)

        # 3. Run FedAvg
        a = schedulers.fedavg(tasks, resources)
        check_and_store('FedAvg', tasks, resources, a, cost)

        # 4. Run Fed-LBAP
        a = schedulers.fed_lbap(tasks, resources, cost)
        check_and_store('Fed-LBAP', tasks, resources, a, cost)

        # 5. Run OLAR
        a = schedulers.olar(tasks, resources, cost, lower_limit, upper_limit)
        check_and_store('OLAR', tasks, resources, a, cost)


def check_and_store(name, tasks, resources, assignment, cost):
    """
    Checks if the results are correct and stores them in the logger.

    Parameters
    ----------
    name : string
        Name of the scheduler
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    assignment : np.array(shape=(resources))
        Assignment of tasks to resources
    cost : np.array(shape=(resources, tasks+1))
        Cost functions per resource (C)
    """
    cmax = support.get_makespan(cost, assignment)
    if support.check_total_assigned(tasks, assignment) == False:
        print(f'-- {name} failed to assign {tasks} tasks to' +
              f' {resources} resources ({np.sum(assignment)}' +
              f' were assigned).')
    logger.store(f'{name},{tasks},{resources},{cmax}')


if __name__ == '__main__':
    run_nlogn_costs()
