#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing quantiles of specific aggregation function for Multivariate Brownian Bridge, by simulation.
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def simulate_brownian_motion(dim, n_simus, n_points=10000):
    """
    Generate multivariate Brownian motion trajectories.
    The size of the multivariate BM is given by "dim",
    the number of simulations (number of Multi-BM generated) by "n_simus"

    :param dim: dimension of the trajectory
    :param n_simus: number of simulated trajectories
    :param n_points: number of points per trajectory
    :return: array of shape (n_simus, n_points,dim) with the  Brownian Motion trajectories
    """

    # some Gaussians (increments of the Brownian motion)
    W = np.random.normal(0, 1, (n_simus, n_points, dim))
    # forcing W_0 = 0 a.s. 
    W[:, 0, :] = 0
    # cumulate the increments to get the Multivariate BM simulations
    Xn = (1 / np.sqrt(n_points)) * np.cumsum(W, 1)

    return Xn


def covariance_matrix(X,y,beta):
    """Compute the Covariance Matrix estimate.
    Eq. 7, pag. 6. Zeileis, 2005 -- Implementing a Class of Structural Change Tests
    paper link: https://www.zeileis.org/papers/Zeileis-2006.pdf"""

    J = np.zeros((beta.shape[0],beta.shape[0]))

    for row_id in range(X.shape[0]):
        Xi = X[row_id].reshape(1, -1)
        intercept_val = np.ones((1, 1))
        Xi = np.c_[intercept_val, Xi]
        yi = np.array(y[row_id], ndmin=2)

        # compute unidim score_vector (for the single unit i)
        xitxi = Xi.T @ Xi
        xitxi_beta = xitxi @ beta
        xityi = Xi.T @ yi
        score_vector_i = xitxi_beta.reshape(-1, 1) - xityi
        J += score_vector_i @ score_vector_i.T

    J = J / X.shape[0]
    return J


def simulate_brownian_bridge(dim, n_simus, n_points=10000, visualize=False):
    """

    Generate multivariate Brownian Bridge trajectories.
    The size of the multivariate BB is given by "dim",
    the number of simulations (number of Multi-BB generated) by "n_simus"

    :param dim: dimension of the trajectory
    :param n_simus: number of simulated trajectories
    :param n_points: number of points per trajectory
    :param visualize: if True, plots each dim of each Multidimensional BB
    :return: array of shape (n_simus, n_points,dim) with the  Brownian Motion trajectories
    """

    Xn = simulate_brownian_motion(dim, n_simus, n_points)
    traj = rescale_trajectories(Xn, visualize=visualize)
    return traj


def rescale_trajectories(Xn, visualize=False):
    """
    Rescale Brownian Motion trajectories, to make it a Brownian Bridge with a=0, b=0

    :param Xn: Brownian Motion trajectories
    :param visualize: if True, plots each dim of each Multidimensional BB
    :return: array of shape (n_simus, n_points,dim) with the  Brownian Motion trajectories
    """
    n_simus = Xn.shape[0]
    n_points = Xn.shape[1]
    dim = Xn.shape[2]

    t_range = (1 / (n_points - 1)) * np.arange(0, n_points)
    aux = np.repeat(t_range.reshape(-1, 1), repeats=dim, axis=1)

    traj = np.zeros(Xn.shape)
    for i_simu in range(n_simus):
        traj[i_simu] = Xn[i_simu] - aux * Xn[i_simu, -1, :]
        # sanity check: show the first dimension of the BB for each simulation
        if visualize:
            for dimension in range(dim):
                plt.plot(t_range, traj[i_simu, :, dimension])
    if visualize:
        plt.show()
    return traj


def compute_bb_quantile(dim, n_simus=2000, n_points=10000, aggregation_function="maxmax", quantile=0.95,
                        visualize=False):
    """
    Here a certain aggregation function is chosen, to reduce a Multivariate Brownian Bridge process
    into a single real value. The distribution of such value is inspected by simulation
    and the result is the requested quantile value of this distribution.

    :param dim: dimension of the trajectory
    :param n_simus: number of simulated trajectories
    :param n_points: number of points per trajectory
    :param aggregation_function: string representing the functional to aggregate the Multivariate BB into a single number
    :param quantile: requested quantile
    :param visualize: if True, plots each dim of each Multidimensional BB and the distribution plot
    :return: real number representing the quantile value
    """

    init_time = time.time()
    # generating some trajectories
    traj = simulate_brownian_bridge(dim, n_simus, n_points, visualize=visualize)

    # for each trajectory, look at some statistic and store everything
    stat_store = np.zeros((n_simus,))
    if aggregation_function == "maxmax":
        stat_store = np.max(np.abs(traj), axis=2)
        stat_store = np.max(stat_store, axis=1)
    elif aggregation_function == "maxsum":
        sum_matrix = np.sum(np.abs(traj), axis=2)
        stat_store = np.max(sum_matrix, axis=1)
    else:
        raise Exception(
            "{} not yet implemented. Please stick with 'maxmax' or 'maxsum' for now".format(aggregation_function))

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        ax.hist(stat_store, bins=100)

    # get the quantile
    bb_quantile = np.quantile(stat_store, quantile)

    end_time = time.time()
    print("Time to run {} Brownian Bridge simulations, {} dims, {} points: {:.2f} sec".format(n_simus, dim, n_points,
                                                                                              end_time - init_time))
    return bb_quantile


if __name__ == "__main__":
    print(compute_bb_quantile(dim=2, n_simus=10000, n_points=50000, visualize=False,
                              aggregation_function="maxsum"))
    # print(compute_bb_quantile(dim=2, n_simus=10000, n_points=50000, visualize=False,
    #                           aggregation_function="maxmax"))
