"""
Module to generate devices with different cost functions.
"""

import numpy as np


# Lower and upper limits of the uniform distribution used for sampling
low_random = 1
high_random = 10


def create_linear_costs(
        rng_seed,
        matrix,
        index,
        tau,
        verbose=False
        ):
    """
    Fills a row of the Cost matrix based on a linear function.

    Parameters
    ----------
    rng_seed : int
        Seed to the random number generator
    matrix : np.ndarray
        Matrix of costs
    index : int
        Row of the matrix to fill
    tau : int
        Size of the row to fill (number of tasks)
    verbose : boolean (default False)
        True if information of the device should be printed

    Notes
    -----
    The linear function is on the format f(x) = a + bx,
    where a and b are randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    np.random.seed(rng_seed)
    # Generates alpha and beta
    alpha, beta = np.random.uniform(low_random, high_random, 2)
    if verbose:
        print(f'[{index}] - Creating linear costs with f(x) =' +
              f' {alpha} + {beta}*x' +
              f'. RNG seed = {rng_seed}')
    # Fills row in the matrix
    matrix[index][:] = [alpha + beta*x for x in range(tau+1)]


def create_quadratic_costs(
        rng_seed,
        matrix,
        index,
        tau,
        verbose=False
        ):
    """
    Fills a row of the Cost matrix based on a quadratic function.

    Parameters
    ----------
    rng_seed : int
        Seed to the random number generator
    matrix : np.ndarray
        Matrix of costs
    index : int
        Row of the matrix to fill
    tau : int
        Size of the row to fill (number of tasks)
    verbose : boolean (default False)
        True if information of the device should be printed

    Notes
    -----
    The quadratic function is on the format f(x) = a + bx + cx^2,
    where a, b, and c are randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    np.random.seed(rng_seed)
    # Generates alpha, beta and gamma
    alpha, beta, gamma = np.random.uniform(low_random, high_random, 3)
    if verbose:
        print(f'[{index}] - Creating quadratic costs with f(x) =' +
              f' {alpha} + {beta}*x + {gamma}*x^2' +
              f'. RNG seed = {rng_seed}')
    # Fills row in the matrix
    matrix[index][:] = [alpha + beta*x + gamma*x*x for x in range(tau+1)]


def create_nlogn_costs(
        rng_seed,
        matrix,
        index,
        tau,
        verbose=False
        ):
    """
    Fills a row of the Cost matrix based on an n log n function.

    Parameters
    ----------
    rng_seed : int
        Seed to the random number generator
    matrix : np.ndarray
        Matrix of costs
    index : int
        Row of the matrix to fill
    tau : int
        Size of the row to fill (number of tasks)
    verbose : boolean (default False)
        True if information of the device should be printed

    Notes
    -----
    The n log n function is on the format f(x) = a + bx log(1+x),
    where a and b are randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    np.random.seed(rng_seed)
    # Generates alpha, beta and gamma
    alpha, beta = np.random.uniform(low_random, high_random, 2)
    if verbose:
        print(f'[{index}] - Creating quadratic costs with f(x) =' +
              f' {alpha} + {beta}*x*log(x)' +
              f'. RNG seed = {rng_seed}')
    # Fills row in the matrix
    matrix[index][:] = [alpha + beta*x*np.log(x+1)
                        for x in range(tau+1)]


def create_recursive_costs(
        rng_seed,
        matrix,
        index,
        tau,
        verbose=False
        ):
    """
    Fills a row of the Cost matrix based on a recursive function.

    Parameters
    ----------
    rng_seed : int
        Seed to the random number generator
    matrix : np.ndarray
        Matrix of costs
    index : int
        Row of the matrix to fill
    tau : int
        Size of the row to fill (number of tasks)
    verbose : boolean (default False)
        True if information of the device should be printed

    Notes
    -----
    The recursive function is on the format f(x) = f(x-1) + a,
    where a is randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    np.random.seed(rng_seed)
    # Generates random values
    values = np.random.uniform(low_random, high_random, tau+1)
    if verbose:
        print(f'[{index}] - Creating random costs with f(x) =' +
              f' f(x-1) + alpha' +
              f'. RNG seed = {rng_seed}')
    # Fills row in the matrix
    matrix[index][:] = np.cumsum(values)
