"""# Description of the experiment:
#
# - We generate the costs to up to 10.000 tasks for 10 and 100 resources.
# All costs follow quadratic functions with RNG seeds [700..799].
# - We schedule from 1.000 to 10.000 tasks in increments of 100.
# - We run OLAR, and the extended versions of Fed-LBAP, FedAvg, and
# Proportional.
# -- For Proportional, we generate three different results using
# three different values for k (1, #tasks/#resources, #tasks)
# - All resources have a lower limit of 4, except the resource with the
# highest cost for the number of tasks. That resource gets a minimum of
# (tasks/resources)/4.
# - All resources have an upper limit of 2*(tasks/resources), except the
# resource with the lowest cost for the number of tasks. That resource
# gets an upper limit of (tasks/resources)/2.
# - Every result is verified and logged to a CSV file.
"""

import numpy as np
import code.support as support
import code.schedulers as schedulers
import code.devices as devices


# File containing the results
logger = support.Logger('results_with_quadratic_costs_and_limits_proportional_only.csv')
rng_seed_resources = 700
min_tasks = 1000
max_tasks = 10001
step_tasks = 1000


def run_quadratic_costs():
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
    # Fills the cost matrix with costs based on a quadratic function
    base_seed = rng_seed_resources
    for i in range(resources):
        devices.create_quadratic_costs(base_seed, cost, i, max_tasks)
        base_seed += 1

    # Iterates over the number of tasks running all schedulers
    for tasks in range(min_tasks, max_tasks, step_tasks):
        # Prepares the upper and lower limit arrays
        avg_tasks = tasks // resources
        lower_limit = np.full(shape=resources, fill_value=4)
        upper_limit = np.full(shape=resources, fill_value=avg_tasks*2)
        # Finds the resource with the maximum and minimum costs for 'tasks'
        max_index = np.argmax(cost[:, tasks])
        lower_limit[max_index] = avg_tasks // 4
        min_index = np.argmin(cost[:, tasks])
        upper_limit[min_index] = avg_tasks // 2

        # 2. Run Proportional with three task values to base itself
        a = schedulers.extended_proportional(tasks, resources, cost, avg_tasks,
                                             lower_limit, upper_limit)
        check_and_store(f'Ext-Proportional-(tasks/res)', tasks, resources, a, cost,
                        lower_limit, upper_limit)
        a = schedulers.extended_proportional(tasks, resources, cost, tasks,
                                             lower_limit, upper_limit)
        check_and_store(f'Ext-Proportional-(tasks)', tasks, resources, a, cost,
                        lower_limit, upper_limit)
        a = schedulers.proportional(tasks, resources, cost, avg_tasks)
        check_and_store(f'Proportional-(tasks/res)', tasks, resources, a, cost,
                        lower_limit, upper_limit)
        a = schedulers.proportional(tasks, resources, cost, tasks)
        check_and_store(f'Proportional-(tasks)', tasks, resources, a, cost,
                        lower_limit, upper_limit)


def check_and_store(name, tasks, resources, assignment, cost,
                    lower_limit, upper_limit):
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
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource
    """
    cmax = support.get_makespan(cost, assignment)
    if support.check_total_assigned(tasks, assignment) == False:
        print(f'-- {name} failed to assign {tasks} tasks to' +
              f' {resources} resources ({np.sum(assignment)}' +
              f' were assigned).')
    if support.check_limits(assignment, lower_limit, upper_limit) == False:
        print(f'-- {name} failed to respect the lower or upper' +
              f' limits when assigning {tasks} tasks to' +
              f' {resources} resources.')
    logger.store(f'{name},{tasks},{resources},{cmax}')


if __name__ == '__main__':
    run_quadratic_costs()
