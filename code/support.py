"""
Module containing support functions for the experiments
"""

import numpy as np
import shutil
from io import StringIO


class Logger:
    """
    Logging class to support the experiments.

    Attributes
    ----------
    filename : string
        Name of the file to write
    log_buffer : StringIO
        Text buffer containing results
    verbosity : boolean
        True if strings should be printed to the standard output too
    """

    def __init__(self, filename, verbosity=False):
        self.filename = filename
        self.log_buffer = StringIO()
        self.verbosity = verbosity

    def header(self, info):
        """
        Stores the header string in the log buffer.

        Parameters
        ----------
        info : string
            String to log
        """
        self.log_buffer.write(info)

    def store(self, info):
        """
        Stores the string in the log buffer.

        Parameters
        ----------
        info : string
            String to log
        """
        self.log_buffer.write(info + '\n')
        if self.verbosity is True:
            print(f'Result: {info}')

    def finish(self):
        """
        Writes the log buffer to a file.
        """
        with open(self.filename, 'w') as logfile:
            self.log_buffer.seek(0)
            shutil.copyfileobj(self.log_buffer, logfile)


def get_makespan(
        cost,
        assignment
        ):
    """
    Computes the makespan of a given assignment of tasks to resources.

    Parameters
    ----------
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    assignment : np.array(shape=(resources))
        Assignment of tasks to resources

    Returns
    -------
    numpy.float64
        Makespan

    Notes
    -----
    The makespan is the maximum cost among all resources based on the
    number of tasks assigned to them.
    """
    # gets the cost for each resource indexed by the assignment array
    cost_by_resource = cost[np.arange(len(cost)), assignment]
    # gets the maximum cost
    makespan = np.max(cost_by_resource)
    return makespan


def check_limits(
        assignment,
        lower_limit,
        upper_limit
        ):
    """
    Checks if any resource has less or more tasks than its limits.

    Parameters
    ----------
    assignment : np.array(shape=(resources))
        Assignment of tasks to resources
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    boolean
        True is the limits are respected
    """
    # np.any returns True if any of the values is True
    # True is in this case happens if any assignments disrespect the limits
    # beware that np.any gives other values for normal arrays (not np.array)
    any_below = np.any(assignment < lower_limit)
    any_above = np.any(assignment > upper_limit)
    return not (any_below or any_above)


def check_total_assigned(
        tasks,
        assignment
        ):
    """
    Checks if the total number of tasks assigned matches the number of tasks.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    assignment : np.array(shape=(resources))
        Assignment of tasks to resources

    Returns
    -------
    boolean
        True is they match
    """
    return tasks == np.sum(assignment)
