"""
Module containing scheduling algorithms.
"""

import numpy as np
import heapq
import code.support as support


def olar(
        tasks,
        resources,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources using OLAR.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    heap = []
    # Assigns lower limit to all resources
    assignment = np.copy(lower_limit)
    for i in range(resources):
        # Initializes the heap
        if assignment[i] < upper_limit[i]:
            heap.append((cost[i][assignment[i]+1], i))
    heapq.heapify(heap)
    # Computes zeta (sum of lower limits)
    zeta = np.sum(lower_limit)
    # Iterates assigning the remaining tasks
    for t in range(zeta+1, tasks+1):
        c, j = heapq.heappop(heap)  # Find minimum cost
        assignment[j] += 1  # Assigns task t
        # Checks if more tasks can be assigned to j
        if assignment[j] < upper_limit[j]:
            heapq.heappush(heap, (cost[j][assignment[j]+1], j))
    return assignment


def fed_lbap(
        tasks,
        resources,
        cost,
        ):
    """
    Finds an assignment of tasks to resources using Fed-LBAP.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources

    Notes
    -----
    Fed-LBAP is presented in
    "Wang, Cong, Xin Wei, and Pengzhan Zhou. "Optimize scheduling of federated
    learning on battery-powered mobile devices." In 2020 IEEE International
    Parallel and Distributed Processing Symposium (IPDPS), pp. 212-221.
    IEEE, 2020."

    Minor changes had to be made to the algorithm presented in the article.
    They include:
    - changing the positions where the median and D' are set (from outside
    to inside the while loop)
    - changes to stop criterion of the main loop (min is always < max
    for some corner cases)
    - accepting Aj = 0
    - making sure D' >= D when the algorithm stops (last valid mapping
    using Cbar[max])
    The algorithm may still assign more tasks than was requested for some
    corner cases.
    """
    assignment = np.zeros(resources, dtype=int)
    # initializes and sorts Cbar
    # ignores Ci(0) and anything over Ci(tau)
    sorted_cost = cost[:, 1:tasks+1].flatten()
    sorted_cost.sort()
    # initializes min and max
    min_index = 0
    max_index = len(sorted_cost)
    still_searching = True
    while still_searching is True:
        # computes median
        median_index = int((max_index + min_index)/2)
        assigned_tasks = 0  # initializes D'
        median_cost = sorted_cost[median_index]  # gets C*
        # loop finding the assignments that respect the median_cost
        for j in range(resources):
            # gets Aj.
            # searchsorted will return the index to the first position
            # with a value larger than median_cost
            index = np.searchsorted(cost[j], median_cost, side='right')
            # the max operator helps here for the case where
            # the  median_cost was larger than cost[j][0] (index = 0)
            assignment[j] = max(0, index-1)
            # updates D'
            assigned_tasks += assignment[j]
        # step in the binary search for the makespan in Cbar
        if assigned_tasks < tasks:
            if min_index == median_index:
                # median will not receive new values, so we are stopping
                # but, before that, we need an assignment that covers all tasks
                still_searching = False
                # loop finding the assignments that respect the median_cost
                median_cost = sorted_cost[max_index]  # gets the final C*
                for j in range(resources):
                    index = np.searchsorted(cost[j], median_cost, side='right')
                    assignment[j] = max(0, index-1)
            else:
                min_index = median_index
        else:
            if max_index == median_index:
                # median will not receive new values, so we are stopping
                # the last solution covers all tasks
                still_searching = False
            else:
                max_index = median_index
    return assignment


def fedavg(
        tasks,
        resources,
        ):
    """
    Finds an assignment of tasks to resources using FedAvg.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources

    Notes
    -----
    FederatedAveraging is presented in
    "McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise
    Aguera y Arcas. "Communication-efficient learning of deep networks from
    decentralized data." In Artificial Intelligence and Statistics,
    pp. 1273-1282. PMLR, 2017."

    The algorithm is cost-oblivious and splits the tasks equality among
    the resources.
    """
    # divides the tasks as equally as possible
    mean_tasks = tasks // resources
    # but it sill may have some leftovers
    leftover = tasks % resources
    assignment = np.full(shape=resources, fill_value=mean_tasks)
    if leftover > 0:
        # adds the leftover to the first resources
        assignment[0:leftover] += 1
    return assignment


def random(
        tasks,
        resources,
        rng_seed
        ):
    """
    Randomly assigns tasks to resources.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    rng_seed : int
        Seed to the random number generator

    Returns
    -------
    np.array(shape=(r))
        Assignment of tasks to resources

    Notes
    -----
    The random algorithm generates a random number uniformly in [1,10)
    for each resource.
    It uses the sum of all random numbers to find the proportion of
    tasks each resource will receive.
    If any tasks are missing due to rounding errors, we randomly add
    them to the resources.
    """
    # prepares the random number generator
    np.random.seed(rng_seed)
    # generates a random number uniformly for each resource
    random_numbers = np.random.uniform(1, 10, resources)
    random_sum = np.sum(random_numbers)
    # splits the tasks using the proportion from the random numbers
    random_tasks = tasks*random_numbers/random_sum
    # transforms them to integers
    assignment = random_tasks.astype(int)
    # assigns any missing tasks
    total_assigned = np.sum(assignment)
    for i in range(total_assigned, tasks):
        random_index = np.random.randint(resources)
        assignment[random_index] += 1
    return assignment


def proportional(
        tasks,
        resources,
        cost,
        k
        ):
    """
    Assigns tasks to resources based on a proportion of the cost functions.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    k : int
        Number of tasks to check the costs and compute a proportion

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources

    Notes
    -----
    The proportional algorithm checks the cost to map 'id_for_proportion'
    tasks to each resource. These costs are used to compute an inverse
    proportion of the number of tasks to be received (the higher the cost,
    the less tasks a resource receives).
    If any tasks are missing due to rounding errors, they are assigned
    one by one to the resources in order.
    """
    # gets the cost for k tasks on all resources
    cost_for_k = cost[:, k]
    # gets the maximum cost to find the proportion
    max_cost = np.max(cost_for_k)
    # finds how many tasks each resource could compute with that cost
    many_tasks = max_cost/cost_for_k
    # gets the sum of this value to compute the proportion to give
    # to each resource
    tasks_sum = np.sum(many_tasks)
    # splits the tasks using the proportion
    proportional_tasks = tasks*many_tasks/tasks_sum
    # transforms them to integers
    assignment = proportional_tasks.astype(int)
    # assigns any missing tasks
    # as the rounding is done at the end, there should be no more than
    # 'resources' tasks missing
    total_assigned = np.sum(assignment)
    leftover = tasks - total_assigned
    if leftover > 0:
        # adds the leftover to the first resources
        assignment[0:leftover] += 1
    return assignment


def extended_fed_lbap(
        tasks,
        resources,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources using an extended
    version of Fed-LBAP.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    assignment = np.zeros(resources, dtype=int)
    # initializes and sorts Cbar
    sorted_cost = []
    for j in range(resources):
        # only adds to Cbar values above the lower limit
        low = lower_limit[j]+1
        # only adds to Cbar values that go up to the upper limit
        up = upper_limit[j]
        sorted_cost = np.append(sorted_cost, cost[j, low:up+1])
    sorted_cost.sort()
    # initializes min and max
    min_index = 0
    max_index = len(sorted_cost)
    still_searching = True
    while still_searching is True:
        # computes median
        median_index = int((max_index + min_index)/2)
        assigned_tasks = 0  # initializes D'
        median_cost = sorted_cost[median_index]  # gets C*
        # loop finding the assignments that respect the median_cost
        for j in range(resources):
            # gets Aj.
            # searchsorted will return the index to the first position
            # with a value larger than median_cost
            index = np.searchsorted(cost[j], median_cost, side='right')
            # the max operator helps here for the case where
            # the  median_cost was larger than cost[j][0] (index = 0)
            # we use the lower and upper limit here to make sure
            # the solution will be viable
            assignment[j] = min(max(lower_limit[j], index-1), upper_limit[j])
            # updates D'
            assigned_tasks += assignment[j]
        # step in the binary search for the makespan in Cbar
        if assigned_tasks < tasks:
            if min_index == median_index:
                # median will not receive new values, so we are stopping
                # but, before that, we need an assignment that covers all tasks
                still_searching = False
                assigned_tasks = 0  # initializes D'
                median_cost = sorted_cost[max_index]  # gets the final C*
                # loop finding the assignments that respect the median_cost
                for j in range(resources):
                    index = np.searchsorted(cost[j], median_cost, side='right')
                    # we use the lower and upper limit here to make sure
                    # the solution will be viable
                    assignment[j] = min(max(lower_limit[j], index-1),
                                        upper_limit[j])
                    # updates D'
                    assigned_tasks += assignment[j]
            else:
                min_index = median_index
        else:
            if max_index == median_index:
                # median will not receive new values, so we are stopping
                # the last solution covers all tasks
                still_searching = False
            else:
                max_index = median_index

    # final verification to fix any cases where more tasks than need end
    # up being assigned
    if assigned_tasks > tasks:
        # fix: remove tasks from the first resource available
        # where this does not lead to a problem with the lower limit of tasks
        excess_tasks = assigned_tasks - tasks
        j = 0
        while excess_tasks > 0:
            if assignment[j] > lower_limit[j]:
                # gets how many tasks can be removed without issue
                tasks_to_remove = min(assignment[j] - lower_limit[j],
                                      excess_tasks)
                # removes tasks
                assignment[j] -= tasks_to_remove
                # updates the excess
                excess_tasks -= tasks_to_remove
                # moves to the next resource
                j += 1

    return assignment


def extended_fedavg(
        tasks,
        resources,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources using an extended
    version of FedAvg.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # divides the tasks as equally as possible
    mean_tasks = tasks // resources
    # but it sill may have some leftovers
    leftover = tasks % resources
    assignment = np.full(shape=resources, fill_value=mean_tasks)
    if leftover > 0:
        # adds the leftover to the first resources
        assignment[0:leftover] += 1
    # extended phase: checks if the assignment is valid
    if support.check_limits(assignment, lower_limit, upper_limit) == True:
        return assignment
    else:
        # the base assignment was invalid, so we switch to a second alternative
        # 1. give all resources their lower_limit
        # 2. iteratively add tasks to the resources with the least tasks that
        # can still receive more

        # Assigns lower limit to all resources
        assignment = np.copy(lower_limit)
        heap = []
        for i in range(resources):
            # Initializes the heap
            if assignment[i] < upper_limit[i]:
                # assignment[i] - mean_tasks: how below a resource is from the
                # mean. The more negative, the higher the priority
                heap.append((assignment[i] - mean_tasks, i))
        heapq.heapify(heap)
        # Computes zeta (sum of lower limits)
        zeta = np.sum(lower_limit)
        # Iterates assigning the remaining tasks
        for t in range(zeta+1, tasks+1):
            c, j = heapq.heappop(heap)  # Find resource with minimum #tasks
            assignment[j] += 1  # Assigns task t
            # Checks if more tasks can be assigned to j
            if assignment[j] < upper_limit[j]:
                heapq.heappush(heap, (assignment[j] - mean_tasks, j))
        return assignment


def extended_proportional(
        tasks,
        resources,
        cost,
        k,
        lower_limit,
        upper_limit
        ):
    """
    Assigns tasks to resources based on an extended version of the
    proportional algorithm.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    k : int
        Number of tasks to check the costs and compute a proportion
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # gets the cost for k tasks on all resources
    cost_for_k = cost[:, k]
    # gets the maximum cost to find the proportion
    max_cost = np.max(cost_for_k)
    # finds how many tasks each resource could compute with that cost
    many_tasks = max_cost/cost_for_k
    # gets the sum of this value to compute the proportion to give
    # to each resource
    tasks_sum = np.sum(many_tasks)
    # splits the tasks using the proportion
    proportional_tasks = tasks*many_tasks/tasks_sum
    # transforms them to integers
    assignment = proportional_tasks.astype(int)
    # assigns any missing tasks
    # as the rounding is done at the end, there should be no more than
    # 'resources' tasks missing
    total_assigned = np.sum(assignment)
    leftover = tasks - total_assigned
    if leftover > 0:
        # adds the leftover to the first resources
        assignment[0:leftover] += 1
    # extended phase: checks if the assignment is valid
    if support.check_limits(assignment, lower_limit, upper_limit) == True:
        return assignment
    else:
        # the base assignment was invalid, so we switch to a second alternative
        # 1. give all resources their lower_limit
        # 2. iteratively add tasks to the resources that are the farthest from
        # the desired proportion can still receive more tasks

        # Assigns lower limit to all resources
        assignment = np.copy(lower_limit)
        heap = []
        for i in range(resources):
            # Initializes the heap
            if assignment[i] < upper_limit[i]:
                # estimated cost of mapping the next task to i
                # based on the proportional cost
                estimated_cost = (assignment[i]+1)*(cost_for_k[i]/k)
                heap.append((estimated_cost, i))
        heapq.heapify(heap)
        # Computes zeta (sum of lower limits)
        zeta = np.sum(lower_limit)
        # Iterates assigning the remaining tasks
        for t in range(zeta+1, tasks+1):
            # Finds resource with minimum estimated cost
            c, j = heapq.heappop(heap)
            assignment[j] += 1  # Assigns task t
            # Checks if more tasks can be assigned to j
            if assignment[j] < upper_limit[j]:
                    estimated_cost = (assignment[j]+1)*(cost_for_k[j]/k)
                    heapq.heappush(heap, (estimated_cost, j))
        return assignment
