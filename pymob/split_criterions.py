import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.linear_model import LinearRegression

def compute_loglik_process(sorted_X, beta, sorted_y, split_indices, aggregation_function, covariance_matrix):

    # compute the multidimensional fluctuation process
    W = compute_W(sorted_X, beta, sorted_y)
    # normalize by the covariance matrix
    efp = fractional_matrix_power(covariance_matrix, -0.5) @ W

    # aggregate into a unidimensional process
    split_scores = aggregate_efp(efp, aggregation_function=aggregation_function, bb_limit=True)

    # keep only feasable splits
    split_scores = [val if id_val in split_indices else -np.inf for id_val, val in enumerate(split_scores)]

    return split_scores


def compute_W(sorted_X, beta, sorted_y):
    '''This function closely follows the Eq.4 in Zeileis 2005 "A Unified Approach to Structural Change Tests..."

        Starting from the loglikelihood definition, we compute its derivative wrt the beta parameters.
        This formula is computed on all the possible left leaves
        (starting from the very first ordered points, adding one point at the time),
        substituting beta with its estimates on the parent leaf.

        We obtain a multidimensional fluctuation process W, of size (n_points X n_vars+1)

        Parameters
        ----------

        '''

    n_points = sorted_X.shape[0]

    # W has shape n*k, for code convenience (it is going to be transposed at return time)
    W = np.zeros((n_points, beta.shape[0]))
    score_vector_cum = np.zeros((beta.shape[0], 1))

    # adding the padding once and for all
    X_w_padding = add_intercept(sorted_X)

    # predictions
    y_hat = np.dot(X_w_padding, beta)

    # maximum likelihood estimator of the variance
    sigma_hat_sq = np.mean(np.square(sorted_y - y_hat))

    for row_id in range(sorted_X.shape[0]):

        Xi = X_w_padding[row_id].T
        yi = np.array(sorted_y[row_id])

        # compute unidim score_vector (for the single unit i)
        xitxi = np.outer(Xi, Xi)
        xitxi_beta = xitxi @ beta
        xityi = yi * Xi
        score_vector_i = xityi - xitxi_beta

        # scale if variance is not zero
        if sigma_hat_sq > np.finfo(float).eps:
            score_vector_i /= sigma_hat_sq

        # compute cumulative score_vector (subgroup S1 until unit i included)
        score_vector_cum = score_vector_cum + score_vector_i.T
        W[row_id] = score_vector_cum[:, 0]

    W = W / np.sqrt(n_points)

    return W.T


def aggregate_efp(efp, aggregation_function="maxsum", bb_limit=False):
    if aggregation_function == "maxsum":

        compressed_efp = list(np.sum(np.abs(efp), axis=0))

        if bb_limit:
            n = efp.shape[1]
            variance = [(i / n) * (1 - i / n) for i in range(1, n + 1)]
            compressed_efp = [safe_division(val, normalization_constant) for val, normalization_constant in
                              zip(compressed_efp, variance)]

    elif aggregation_function == "maxmax":
        compressed_efp = list(np.max(np.abs(efp), axis=0))
    else:
        raise Exception("Aggregation function not recognized")

    return compressed_efp


def add_intercept(sorted_X):
    """adding the padding, i.e. the 1s column of the intercept in the design matrix"""

    num_samples = sorted_X.shape[0]
    padding = np.ones((num_samples, 1))
    X_tilde = np.c_[padding, sorted_X]
    return X_tilde


def safe_division(n, d):
    """Custom division to allow division by 0
    (the specific value won't be considered in the future, since they are at the boundaries,
    hence the relative splitpoint is unfeasable """
    return n / d if d else 0


def compute_mse_process_old(sorted_X, beta, sorted_y, cut_index, num_samples_node):
    '''Uses the Derivative of the Mean Square Error (MSE) function, wrt beta.
        Since MSE is proportional to R^2, the argmax stays the same.
        The formula is evaluated for the beta estimates of the Regression on the entire parent node,
        while the X,y are the left child units for the currently considered split.
        Formula: (XT * X) * beta - (XT * y)    shape: (num_variables,1) '''

    x_slice = sorted_X[:cut_index]
    y_slice = sorted_y[:cut_index]

    # add a column of ones
    num_samples = x_slice.shape[0]
    ones = np.ones((num_samples, 1))
    x_slice = np.c_[ones, x_slice]
    # reshape the y to have (n,1) shape
    y_slice = y_slice[:, np.newaxis]

    score_vector = np.matmul(np.matmul(x_slice.T, x_slice), beta).reshape(-1, 1) - np.matmul(x_slice.T, y_slice)

    score_value = np.sum(np.absolute(score_vector))  # * partition_discount(len(x_slice), num_samples_node)

    return score_value

def compute_mse_process_online(sorted_X, beta, sorted_y, split_indices,
                               aggregation_function):

    num_samples, num_variables = sorted_X.shape
    # initialize the list of scores for each split
    split_scores = np.full((num_samples,), np.NINF)

    # initialize quantities for the online mse process calculation
    cut_index_old = 0
    xtx_old = np.zeros((num_variables + 1, num_variables + 1))
    xty_old = np.zeros((num_variables + 1, 1))

    for cut_index in split_indices:
        # split_scores[cut_index] = score_function(sorted_X, beta, sorted_y, cut_index, num_samples)
        split_scores[cut_index], xtx_old, xty_old = compute_mse_process_online_primitive(sorted_X, beta, sorted_y, cut_index,
                                                                               xtx_old, xty_old, cut_index_old,
                                                                               aggregation_function=aggregation_function)
        cut_index_old = cut_index
    return split_scores


def compute_mse_process_online_primitive(sorted_X, beta, sorted_y, cut_index, xtx_old, xty_old, cut_index_old,
                               aggregation_function):
    """
    Use the Derivative of the Mean Square Error (MSE) function, wrt beta.
        Since MSE is proportional to R^2, the argmax stays the same.
        The formula is evaluated for the beta estimates of the Regression on the entire parent node,
        while the X,y are the left child units for the currently considered split.
        The Formula is: (XT * X) * beta - (XT * y)    shape: (num_variables,1)

        We find a way to do it in an incremental way:
        storing (XT * X) and (XT * y) from the previous iteration,
        and adding the newlines to obtain the new (XT * X) and (XT * y) given by the formulas:
        (XT * X)_new = (XT * X)_old + x_newlines.T * x_newlines
        (XT * y)_new = (XT * y)_old + x_newlines.T * y_newlines
    """

    newlines = sorted_X[cut_index_old:cut_index]
    ones = np.ones((cut_index - cut_index_old, 1))
    newlines = np.c_[ones, newlines]
    ntn = newlines.T @ newlines
    xtx_new = xtx_old + ntn
    xtx_beta = xtx_new @ beta
    new_y = sorted_y[cut_index_old:cut_index]
    new_y = new_y[:, np.newaxis]
    xty_new = xty_old + newlines.T @ new_y
    score_vector = xtx_beta.reshape(-1, 1) - xty_new

    # IMPORTANT: we do not rescale for the variance \hat{\sigma**2} (Eq. 6 in our paper),
    # because it is a constant for the entire process

    score_value = aggregate_efp(score_vector, aggregation_function=aggregation_function, bb_limit=False)[0]

    return score_value, xtx_new, xty_new


def compute_initial_quantities(X_tilde, sorted_y):
    C_t = X_tilde.T @ X_tilde
    Cinv_t = np.linalg.pinv(C_t)
    beta_t = np.dot(Cinv_t @ X_tilde.T, sorted_y)
    residuals_t = sorted_y - np.dot(X_tilde, beta_t)
    sum_y = np.sum(sorted_y)
    sum_y_sq = np.sum(sorted_y ** 2)

    t = X_tilde.shape[0]
    deviance = sum_y_sq - t * (sum_y / t) ** 2
    r_sq = 1 - np.sum(np.square(residuals_t)) / deviance

    return Cinv_t, beta_t, residuals_t, sum_y, sum_y_sq, r_sq


def compute_online_r2(sorted_X, sorted_y, split_indices, weighted=True):

    num_samples, num_variables = sorted_X.shape

    # computing R2 on both sides (the left and right leaves), by reverting the array
    left_r_sq = compute_online_r2_primitive(sorted_X, sorted_y, side="left")
    right_r_sq = compute_online_r2_primitive(sorted_X, sorted_y, side="right")

    if weighted:
        left_weights = np.arange(1, num_samples, dtype="int")
        right_weights = np.arange(num_samples - 1, 0, -1, dtype="int")
        split_scores = 1 / num_samples * (left_r_sq*left_weights + right_r_sq*right_weights)
    else:
        split_scores = (left_r_sq + right_r_sq) / 2

    # keep only feasable splits
    split_scores = [val if id_val in split_indices else -np.inf for id_val, val in enumerate(split_scores)]


    return split_scores

def compute_online_r2_primitive(sorted_X, sorted_y, side="left"):
    """
    Computing the R2 online.

    :param:
    X: n x d array
    vecy: n array

    :return:
    """

    # getting the shape of the problem
    n_points, dim = sorted_X.shape

    if side == "right":
        sorted_X = np.flip(sorted_X, 0)
        sorted_y = np.flip(sorted_y, 0)

    # adding the padding once and for all
    X_tilde = add_intercept(sorted_X)

    # initialize everything

    # the possible splits are one less than the n_points (eg. with 20 points, we can try out 19 possible splits, not 20)
    r_sq_store = np.zeros((n_points - 1,))
    residuals = np.zeros((n_points - 1,))
    sum_y = 0.0
    sum_y_sq = 0.0

    # compute the initial quantities (they are meaningful when we have at least dim+1 elements in the possible split)
    t = dim + 1
    C_t = X_tilde[:t].T @ X_tilde[:t]
    Cinv_t = np.linalg.pinv(C_t)
    beta_t = np.dot(Cinv_t @ X_tilde[:t].T, sorted_y[:t])
    residuals[:t] = sorted_y[:t] - np.dot(X_tilde[:t], beta_t)
    sum_y = np.sum(sorted_y[:t])
    sum_y_sq = np.sum(sorted_y[:t] ** 2)
    deviance = sum_y_sq - t * (sum_y / t) ** 2
    r_sq = 1 - np.sum(np.square(residuals[:t])) / deviance
    r_sq_store[t - 1] = r_sq

    # main loop
    for t in range(dim + 2, n_points):
        # next point data
        y_t_plus_one = sorted_y[t - 1]
        x_tilde_t_plus_one = X_tilde[t - 1]

        # update variance estimate
        sum_y += y_t_plus_one
        sum_y_sq += y_t_plus_one ** 2
        deviance = sum_y_sq - t * (sum_y / t) ** 2

        # compute key quantities
        aux_1 = np.dot(Cinv_t, x_tilde_t_plus_one)
        aux_2 = 1 + np.dot(x_tilde_t_plus_one.T, aux_1)
        aux_3 = y_t_plus_one - np.dot(x_tilde_t_plus_one.T, beta_t)
        aux_4 = aux_3 / aux_2

        # updating the residuals
        residuals[:t - 1] = residuals[:t - 1] - aux_4 * np.dot(X_tilde[:t - 1], aux_1)
        residuals[t - 1] = aux_4

        # compute the R2
        r_sq = 1 - np.sum(np.square(residuals[:t])) / deviance
        r_sq_store[t - 1] = r_sq

        # update numerator quantities
        Cinv_t = Cinv_t - (1 / aux_2) * np.outer(aux_1, aux_1)
        beta_t = np.dot(np.eye(dim + 1) - (1 / aux_2) * np.outer(aux_1, x_tilde_t_plus_one),
                        beta_t + y_t_plus_one * aux_1)

    if side == "right":
        r_sq_store = np.flip(r_sq_store, 0)

    return r_sq_store

def compute_naive_r2(sorted_X, sorted_y, split_indices):

    num_samples, num_variables = sorted_X.shape
    # initialize the list of scores for each split
    split_scores = np.full((num_samples,), np.NINF)

    for cut_index in split_indices:
        split_scores[int(cut_index)] = compute_naive_r2_primitive(sorted_X, sorted_y, int(cut_index))

    return split_scores

def compute_naive_r2_primitive(X, y, cut_index):
    '''Given a split (index of where the cut should happen in the ordered X,y samples),
     fit a model on left and right partitions and return obj function.

     The objective function should always be minimized in Mob!
     '''

    # # cambio il modello in Huber Regression per regressione robusta a outliers (uso default epsilon di sklearn)
    model = LinearRegression()

    X_left = X[:cut_index]
    y_left = y[:cut_index]
    X_right = X[cut_index:]
    y_right = y[cut_index:]

    # fit left and right model
    model_left = model.fit(X_left,y_left)
    left_r_sq = model_left.score(X_left,y_left)
    model_right = model.fit(X_right,y_right)
    right_r_sq = model_right.score(X_right,y_right)

    return (left_r_sq + right_r_sq)/2
