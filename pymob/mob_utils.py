import warnings

import numpy as np

from .brownian_bridge_utils import covariance_matrix
from .classes import Node, OrderedSplit
from .split_criterions import (compute_loglik_process,
                               compute_mse_process_online, compute_naive_r2,
                               compute_online_r2)


def check_parameters(minsplit: int, n_dims: int, ml_continuous: bool, max_outliers: int) -> None:
    """
    Control that parameters have been initialized properly.
    In particular,

    :param minsplit: minimum number of points in each terminal node
    :param n_dims: number of variables on which to train MOB
    :param ml_continuous: DEPRECATED boolean flag, if True MOB fits simple Linear Regression,
        if False MOB uses Huber Regression and tries to find outliers (and move the split-point to avoid outliers).
    :param max_outliers: DEPRECATED maximum num of outliers in the partition to enable the split-point moving logic.
        Active only if ml_continuous=False
    :return: None
    """

    if minsplit < n_dims + 1:
        warnings.warn("""Minsplit should be greater than the number of features.
            'minsplit' changed to number of variables +1.
             
             It is however suggested to set 'minsplit' at least 10% greater than number of features,
              to obtain reliable R2 values. Consider change the 'minsplit' value""")

    if not ml_continuous:
        if minsplit < max_outliers + 2:
            raise Exception("""Minimum number of units in each leaf should be at least "max_outliers" + 3
             (to guarantee the Regression to be able to run).
             "max_outliers" will be changed to "minsplit" - 3""")


def mob_fit_setup_node(mob_parameters: object, X_subset: np.array, y_subset: np.array, weights: np.array, r2: float,
                       model: object, stopping_rule: object) -> Node:
    """
    Set up a new node of the MOB tree

    :param mob_parameters: Data instance, containing relevant parameters
    :param X_subset: {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples ended up in the current node.
    :param y_subset: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression), ended up in the current node.
    :param weights: array-like, shape (n_samples,)
        Sample weights determining samples ended up in the current node
    :param r2: R2 value of the Linear model on the current node
    :param model: Linear model fitted on input samples in the current node
    :param stopping_rule: either R2StoppingCriterion or BBStoppingCriterion instance
        (depending on the 'method' attribute), defining the stopping criteria
    :return: Node instance
    """

    # compute local standard deviations for the current node
    st_devs = [np.std(X_subset, axis=0), np.std(y_subset)]

    # Pre-fit stopping condition
    if stopping_rule.pre_fit(weights=weights, statistic=r2):
        node = Node(None, weights=weights, terminal=True, leaf_points=[X_subset, y_subset], st_devs=st_devs)
        node.weights = None
        return node

    # Workhorse Function
    splitpoint, best_score = mob_fit_test(X_subset, y_subset, model, mob_parameters.minsplit,
                                          method=mob_parameters.method,
                                          aggregation_function=mob_parameters.aggregation_function)

    # Post Fit Stopping Condition
    if stopping_rule.post_fit(statistic=best_score):
        node = Node(None, weights=weights, terminal=True, leaf_points=[X_subset, y_subset], st_devs=st_devs)
        node.weights = None
        return node
    else:
        node = Node(None, weights, terminal=False, parent_split=splitpoint, st_devs=st_devs)
    node.weights = None

    return node


def mob_fit_test(X: np.array, y: np.array, model: object, minsplit: int, method: str,
                 aggregation_function: str = "maxsum") -> tuple:
    """
    Calculates the best split for the current node.
    FOr each variable separately, evaluate the best splitpoint and the relative score,
    then compare the scores on different variables and select the splitpoint with highest score.
    Returns an object of the class OrderedSplit, containing both the variable_id and the splitpoint value,
    and the related best score

    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    :param y: array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).
    :param model: Linear model fitted on input samples in the current node
    :param minsplit: minimum number of points in each terminal node
    :param method: one of the following 'MSE fluc process','naive R2','online R2','weighted online R2',
        'loglik fluc process'. Defines the MOB splitting criterion.
    :param aggregation_function: one of the following 'maxmax', 'maxsum'.
        Specifies how to aggregate a multidimensional process in the best split quest.
    :return: a tuple containing an object of the class OrderedSplit and the relative score of the split (flaot)
    """

    num_variables = X.shape[1]

    scores = np.zeros((num_variables,))
    splitpoints = np.zeros((num_variables,))
    beta = None
    J = None

    if method in ["loglik fluc process", "MSE fluc process"]:
        beta = get_regression_coef(model)
    if method == "loglik fluc process":
        J = covariance_matrix(X, y, beta)

    for id_var in range(num_variables):
        # get score and splitpoint for the best partition on the given variable
        splitpoint, score = get_best_splitpoint(id_var, X, y, minsplit, beta, method=method, cov_matrix=J,
                                                aggregation_function=aggregation_function)
        splitpoints[id_var] = splitpoint
        scores[id_var] = score

    # get best splitpoint overall
    best_score = np.max(scores)
    best_var_id = np.argmax(scores)
    best_splitpoint = splitpoints[best_var_id]

    split = OrderedSplit(variable_id=best_var_id, split_point=float(best_splitpoint))

    return split, best_score


def get_best_splitpoint(id_var: int, X: np.array, y: np.array, minsplit: int, beta: np.array, method: str,
                        cov_matrix: np.array, aggregation_function: str) -> tuple:
    """
    Find the best splitpoint on a given variable, computed on the X,y arrays

    :param id_var: id of the variable on which to look for the best split
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    :param y: array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).
    :param minsplit: minimum number of points in each terminal node
    :param beta: array-like, shape (num_variables + 1,)
        The Maximum Likelihood (OLS) parameters for the Linear model trained on the data points of the current node,
        contains also the intercept.
    :param method: one of the following 'MSE fluc process','naive R2','online R2','weighted online R2',
        'loglik fluc process'. Defines the MOB splitting criterion.
    :param cov_matrix: array-like, shape (num_variables, num_variables)
        The covariance matrix of the process scores.
    :param aggregation_function: one of the following 'maxmax', 'maxsum'.
        Specifies how to aggregate a multidimensional process in the best split quest.
    :return: a tuple containing the value of the best splitpoint (float) and the related score value (float)
    """

    num_samples, num_variables = X.shape

    # select column and sort it
    var = X[:, id_var]
    sorting_ids = np.argsort(var)
    sorted_vals = var[sorting_ids]
    # calculate the indices of the first occurrences of the unique vals on the given variable
    _, unique_indices = np.unique(sorted_vals, return_index=True)

    # sort arrays, ascending order of the chosen var
    sorted_X = X[sorting_ids]
    sorted_y = y[sorting_ids]

    # # get valid split indices
    # keep equal values in the same partition, consider partitions only with more than minsplit units
    cond = np.logical_and(unique_indices >= minsplit, unique_indices <= num_samples - minsplit)
    split_indices = unique_indices[cond]

    # compute scores of the process, given the optimisazion method
    if method == "MSE fluc process":
        split_scores = compute_mse_process_online(sorted_X, beta, sorted_y, split_indices, aggregation_function)

    elif method == "naive R2":
        split_scores = compute_naive_r2(sorted_X, sorted_y, split_indices)

    elif method == "online R2":
        split_scores = compute_online_r2(sorted_X, sorted_y, split_indices, weighted=False)

    elif method == "weighted online R2":
        split_scores = compute_online_r2(sorted_X, sorted_y, split_indices, weighted=True)

    elif method == "loglik fluc process":
        split_scores = compute_loglik_process(sorted_X, beta, sorted_y, split_indices, aggregation_function, cov_matrix)

    # save the best score and splitpoint for the variable (min, argmin)
    best_cutoff_id = np.argmax(split_scores)
    best_score = split_scores[best_cutoff_id]
    best_splitpoint = sorted_vals[best_cutoff_id]

    return best_splitpoint, best_score


def get_regression_coef(model: object) -> np.array:
    """
    Extract beta coefficients from trained Linear model
    :param model: trained linear model object
    :return: array-like, shape (num_variables + 1,)
        The Maximum Likelihood (OLS) parameters for the Linear model trained on the data points of the current node,
        contains also the intercept.
    """

    intercept = model.intercept_
    coefficients = model.coef_
    beta = np.append(intercept, coefficients)
    return beta


def mob_fit_childweights(node: Node, X: np.array, weights: np.array) -> tuple:
    """
    Determine which of the data samples go into the left and right children of the given node.
    Return two separate arrays of weights: leftweights and rightweights,
    with values different from 0 when the data point ends up in the left or right child respectively

    :param node: the Node object corresponding to the current node
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples ended up in the current node.
    :param weights: array-like, shape (n_samples,)
        Sample weights determining samples ended up in the current node
    :return: tuple containing the leftweights and rightweights arrays
    """

    splitting_column = X[:, node.parent_split.variable_id]

    leftweights = (splitting_column < node.parent_split.split_point) * weights
    rightweights = (splitting_column >= node.parent_split.split_point) * weights

    return leftweights, rightweights


def get_leaves(node: Node) -> np.array:
    """
    Starting from the current 'node', retrieve all the granchildren nodes which are terminal.
    This is a recursive function

    :param node: the Node object corresponding to the current node
    :return: the 1d array containing all the grandchildren terminal nodes
    """

    if node.terminal:
        return node
    ll = get_leaves(node.left_child)
    rr = get_leaves(node.right_child)
    terminal_nodes = np.r_[ll, rr]
    return terminal_nodes


def get_leaf_weights(tree: Node, data: np.array) -> dict:
    """
    Get the weight of each leaf based on the X data passed, i.e. the percentage of data points ending up in each leaf

    :param tree: the initial Node object of the MOB model, containing all the grandchildren
        (MOB must be fitted to have a valid 'tree' attribute)
    :param data: {array-like, sparse matrix}, shape (n_samples, n_features), accepts also pd.DataFrame
            The data samples on which to get the leaves weights.
    :return: a dictionary of the shape {leaf_id: leaf_weight}
    """

    id_leaf_data = list()
    for row in data:
        leaf = get_pred_node(tree, row)
        id_leaf_data.append(leaf.id)

    active_leaves_id, data_count = np.unique(id_leaf_data, return_counts=True)
    leaf_iterator = zip(active_leaves_id, data_count / data.shape[0])
    leaf_weights = {leaf_id: leaf_count for leaf_id, leaf_count in leaf_iterator}

    return leaf_weights


def get_node_preds(tree: Node, data: np.array) -> list:
    """
    Retrieve the prediction of the terminal node related to each row of the data

    :param tree: the initial Node object of the MOB model, containing all the grandchildren
        (MOB must be fitted to have a valid 'tree' attribute)
    :param data: {array-like, sparse matrix}, shape (n_samples, n_features), accepts also pd.DataFrame
            The data samples to be predicted.
    :return: a list of predictions
    """

    preds = list()
    for row in data:
        leaf = get_pred_node(tree, row)
        local_regression = leaf.regression
        pred = local_regression.predict(row.reshape(1, -1))[0]
        preds.append(pred)
    return preds


def get_pred_node(tree: Node, x: np.array) -> Node:
    """
    Find out the terminal node in which the x sample falls.
    This is a recursive function

    :param tree: the initial Node object of the MOB model, containing all the grandchildren
        (MOB must be fitted to have a valid 'tree' attribute)
    :param x: {array-like, sparse matrix}, shape (n_features,), accepts also pd.Series or pd.DataFrame
            The single sample to be predicted.
    :return:
    """

    if tree.terminal:
        return tree
    split = tree.parent_split
    cutpoint = split.split_point
    split_var = split.variable_id

    if x[split_var] < cutpoint:
        return get_pred_node(tree.left_child, x)
    else:
        return get_pred_node(tree.right_child, x)


def get_most_crowded_leaf(mob: object, data: np.array) -> Node:
    """
    Get the leaf with the highest number of datapoints from the data array

    :param mob: fitted object of the MOB class
    :param data: {array-like, sparse matrix}, shape (n_samples, n_features)
            The data samples on which to compute the most crowded leaf.
    :return: a Node object corresponding to the terminal node with the most of the data sample points
    """

    leaf_weights = get_leaf_weights(mob.tree, data)
    max_leaf_id = max(leaf_weights, key=leaf_weights.get)
    leaves_ids = [leaf.id for leaf in mob.leaves]
    max_leaf = [leaf for leaf_id, leaf in zip(leaves_ids, mob.leaves) if leaf_id == max_leaf_id][0]
    return max_leaf


def check_domain(data: np.array, node: Node, verbose: bool = False) -> list:
    """
    Check if samples of the data array fall inside the domain of the given node

    :param data: {array-like, sparse matrix}, shape (n_samples, n_features), accepts also pd.DataFrame
            The data samples to be predicted.
    :param node: a Node object, to test whether data samples is inside its boundaries
        (Might be also the original node (tree) to check whether any sample in data is outside the MOB training domain)
    :param verbose: boolean flag, if True prints a warning
    :return:
    """

    domain = node.domain
    samples_ids_in_domain = list()

    for id_sample, sample in enumerate(data):
        if domain.contains(sample):
            samples_ids_in_domain.append(id_sample)
        else:
            if verbose:
                warnings.warn(f"point {sample} not in domain")
    return samples_ids_in_domain
