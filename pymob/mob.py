"""
MOB Model Regressor (scikit-learn like)
"""

from functools import cached_property

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .brownian_bridge_utils import compute_bb_quantile
from .classes import (BBStoppingCriterion, Data, Node, ProgressBar,
                     R2StoppingCriterion, RealDomain, RealInterval)
from .mob_utils import (check_domain, check_parameters, get_leaves,
                       get_node_preds, mob_fit_childweights,
                       mob_fit_setup_node)
from utils import save_pickle_object


class ParametersWarning(Warning):
    pass


class MOB(BaseEstimator):
    """MOB model Class.
    The class provides a sklearn like estimator, whose behaviour is to recursively partition the input space
    and fit a Linear Model inside each partition.

    The inner logic is similar to the one of Decision Trees, but with a different splitting criterion,
    which optimizes the linear fit in the partitions. """

    def __init__(self, minsplit: int = 20, method: str = 'MSE fluc process', aggregation_function: str = "maxsum",
                 verbose: bool = False,
                 stopping_value: float = None, quantile: float = None, ml_continuous: bool = True,
                 max_outliers: int = 5, domain_dict: dict = None) -> None:

        """

        :param minsplit: minimum number of points in each terminal node
        :param method: one of the following 'MSE fluc process','naive R2','online R2','weighted online R2',
            'loglik fluc process'. Defines the MOB splitting criterion. 
        :param aggregation_function: one of the following 'maxmax', 'maxsum'.
            Specifies how to aggregate a multidimensional process in the best split quest.
        :param verbose: boolean flag to print MOB info
        :param stopping_value: R2 value threshold, when a node obtains R2 greater than stopping_value
            it is considered terminal. Active if method is any apart from 'loglik fluc process'
        :param quantile: DEPRECATED Brownian Bridge quantile,
            which determines the stopping criterion alternative to R2 stopping_value. 
            Active only if method='loglik fluc process'
        :param ml_continuous: DEPRECATED boolean flag, if True MOB fits simple Linear Regression,
            if False MOB uses Huber Regression and tries to find outliers (and move the split-point to avoid outliers).
            Should improve the space partitioning for non-continuous models
            (Tree-based models like Random Forest, Gradient Boosting etc.), but it does not work properly
        :param max_outliers: DEPRECATED maximum num of outliers in the partition to enable the split-point moving logic.
            Active only if ml_continuous=False
        :param domain_dict: Dictionary containing the Input Space boundaries
            (one dict entry per each variable of shape: {'variable_name': RealDomain} or {'variable_name': [min,max]}
        """

        self.minsplit = np.ceil(abs(minsplit))
        self.verbose = verbose
        self.stopping_value = stopping_value
        self.method = method
        self.aggregation_function = aggregation_function
        self.max_outliers = max_outliers
        self.domain_dict = domain_dict
        self.ml_continuous = ml_continuous
        self.quantile = quantile

        self.n_nodes = 1
        self.weights = None
        self.tree = None
        self.is_fitted_ = False
        self.bb_quantile = None

    def _get_stopping_criterion(self):
        """
        Check validity of the 'method' attribute,
        Assign the correct stopping criterion, based on the 'method',
        Check 'quantile','stopping_value' attributes (set defaults if None), compute bb_quantile (if needed).
        All the necessary quantities are saved as class attributes
        
        :return: None
        """

        if self.method == "loglik fluc process":
            self.stopping_criterion = "BB Quantile"
            if not self.quantile:
                self.quantile = 0.95
                self.bb_quantile = compute_bb_quantile(dim=len(self.domain_dict.keys()) + 1, n_simus=2000,
                                                       n_points=20000,
                                                       aggregation_function=self.aggregation_function,
                                                       quantile=self.quantile)
        elif self.method in ["MSE fluc process", "naive R2", "online R2", "weighted online R2"]:
            self.stopping_criterion = "R2"
            if not self.stopping_value:
                self.r2_stopping_value = 0.9
        else:
            raise Exception(f"method {self.method} not recognized as valid choice. "
                            f"Please comply with one of the following:"
                            f"'MSE fluc process','loglik fluc process','naive R2','online R2','weighted online R2'")

    def store_mob_params(self) -> Data:
        """
        Stores relevant MOB attributes as an instance of the Data class

        :return: Data instance, containing relevant parameters
        """
        mob_parameters = Data(minsplit=self.minsplit, verbose=self.verbose, stopping_value=self.stopping_value,
                              method=self.method, aggregation_function=self.aggregation_function,
                              max_outliers=self.max_outliers, domain_dict=self.domain_dict,
                              ml_continuous=self.ml_continuous, quantile=self.quantile, bb_quantile=self.bb_quantile,
                              stopping_criterion=self.stopping_criterion)
        return mob_parameters

    @cached_property
    def leaves(self):
        """Retrieves a list containing leaves of the final tree if fitted, otherwise an empty list"""
        return get_leaves(self.tree) if self.is_fitted_ else list()

    @cached_property
    def variables(self):
        """Returns a list containing the variables names"""
        return self.tree.domain.get_variables()

    def fit(self, X: np.array, y: np.array, weights: np.array = None) -> object:
        """
        Fit the model according to the given training data.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        :param y: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        :param weights: array-like, shape (n_samples,)
            Weights assigned to each data point, for weighted regression.
            By default, MOB assigns value 1 to each data point, i.e. equal importance
        :return: fitted estimator
        """

        # check consistence of the X,y df for sklearn. Convert df to np.arrays
        X, y = check_X_y(X, y, accept_sparse=True)

        # store standard deviations of the variables
        self.global_st_devs = [np.std(X, axis=0), np.std(y)]

        n_points, n_dims = X.shape
        # check consistence of the minsplit parameter
        check_parameters(minsplit=self.minsplit, n_dims=n_dims, ml_continuous=self.ml_continuous,
                         max_outliers=self.max_outliers)

        # set weights vector
        if weights is None:
            self.weights = np.ones([n_points, ])
        else:
            self.weights = weights

        # initialize domain
        if self.domain_dict is None:
            self.domain_dict = {x: RealInterval(bounds=(np.min(X[:, x]), np.max(X[:, x])),
                                                included=(True, True)) for x in range(X.shape[1])}
        full_domain = RealDomain(self.domain_dict)

        # initialize the stopping criterion and stopping_rules
        self._get_stopping_criterion()
        if self.stopping_criterion == "R2":
            stopping_rule = R2StoppingCriterion(minsplit=self.minsplit, verbose=self.verbose,
                                                stopping_value=self.stopping_value)
        elif self.stopping_criterion == "BB Quantile":
            stopping_rule = BBStoppingCriterion(minsplit=self.minsplit, verbose=self.verbose, quantile=self.quantile,
                                                bb_quantile=self.bb_quantile)
        else:
            raise Exception("Non admissible stopping criterion")

        # store parameters as Data class instance
        mob_parameters = self.store_mob_params()

        # initialize progress bar to show approximate ETA
        pb = ProgressBar(n_points)

        # run the recursive workhorse function to fit the tree
        self.tree = self.mob_fit(X, y, self.weights, full_domain, stopping_rule, mob_parameters, pb=pb)
        self.is_fitted_ = True

        pb.close()

        return self

    def mob_fit(self, X: np.array, y: np.array, weights: np.array, domain: RealDomain,
                stopping_rule: R2StoppingCriterion, mob_parameters: Data, split_var: int = None,
                pb: ProgressBar = None) -> Node:
        """
        Recursive function to build the tree.
        This function is called at each node, to find out the best splitting variable and splitpoint.

        The weights parameter is an array of shape (n_samples,), containing only 0 and 1:
        0 when the given sample is not included in the current node, 1 when it is included.
        Using this structure, we pass the entire X,y arrays to each mob_fit instance,
        and we select the units to consider through the weights.

        For future developments, it is also possible to modify the weights value to be in the range [0,1],
        to give more or less importance to certain samples (which may be more important to be correctly predicted).


        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        :param y: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        :param weights: array-like, shape (n_samples,)
            Sample weights to fit on samples of a specific partition
        :param domain: RealDomain instance containing the Input Space boundaries
        :param stopping_rule: either R2StoppingCriterion or BBStoppingCriterion instance
            (depending on the 'method' attribute), defining the stopping criteria
        :param mob_parameters: Data instance, containing the relevant MOB attributes
        :param split_var: id of the variable where the previous split has been performed
        :param pb: ProgressBar instance
        :return: Node instance
        """

        # subsample the dataset to keep only samples in the current partition
        positive_weights = weights > 0

        # initialize the Linear Model to be fitted in the current node
        if self.ml_continuous:
            model = LinearRegression(normalize="deprecated", copy_X=False)
        else:
            model = HuberRegressor(alpha=0.0, max_iter=1000)

        # fit the model in the current node
        try:
            model.fit(X[positive_weights], y[positive_weights])
            r2 = model.score(X[positive_weights], y[positive_weights])

            # if model is Huber, check outliers
            if not self.ml_continuous:
                outliers = model.outliers_
                num_outliers = np.sum(outliers)
                self.verbose and print(f"{num_outliers} outliers out of {np.shape(outliers)} samples\n R2:{r2}")

                # remove outliers if less than max_outliers, and train Linear Regression
                if num_outliers <= self.max_outliers:
                    # remove outliers from partition
                    weights[positive_weights] = weights[positive_weights] - outliers
                    positive_weights = weights > 0

                    # train LinearRegression without outliers
                    model = LinearRegression()
                    model.fit(X, y, sample_weight=weights)
                    r2 = model.score(X, y, sample_weight=weights)

        # logic when the Linear model fit fails
        except Exception as e:
            print("""The Linear Model fitted in the {} node had a non zero output.
                    The node is flagged as terminal, its regression property will contain the Linear model with errors.
                    Hereafter the raised Exception:
                    {}""".format(self.n_nodes + 1, e))
            node = Node(self.n_nodes, weights, leaf_points=[X[positive_weights], y[positive_weights]])
            node.regression = model
            node.st_devs = [np.std(X[positive_weights], axis=0), np.std(y[positive_weights])]
            node.terminal = True
            self.n_nodes += 1
            return node

        # set up the current node
        node = mob_fit_setup_node(mob_parameters, X[positive_weights], y[positive_weights], weights, r2, model,
                                  stopping_rule)
        node.split_var = split_var
        node.domain = domain
        node.id = self.n_nodes
        self.n_nodes += 1
        node.regression = model
        node.score = r2

        if self.verbose:
            split_info = f"Splitting Var={node.parent_split.variable_id}, " \
                         f"Split Point={node.parent_split.split_point:.2f}" if node.parent_split else "Terminal Node"
            print(
                f"Node Id:{node.id}, Node R2={node.score:.2f}, " + split_info + "\n" + f"{node.domain=}" + "\n")

        # split the node in left and right child (if valid splitpoint: splitting variable st_dev must not be 0)
        # may happen if all the points in the node have the same X values (no valid splits)
        if not node.terminal and node.st_devs[0][node.parent_split.variable_id] > 1e-12:
            # compute children size
            leftweights, rightweights = mob_fit_childweights(node, X, weights)
            # get the children domains from the parent domain
            parent_split = node.parent_split
            left_domain, right_domain = node.domain.split(parent_split.split_point, parent_split.variable_id)
            # delete the model if the node is not a leaf
            node.regression = None
            del model

            # recursive call
            node.left_child = self.mob_fit(X, y, leftweights, left_domain, stopping_rule, mob_parameters,
                                           split_var=parent_split.variable_id, pb=pb)
            node.right_child = self.mob_fit(X, y, rightweights, right_domain, stopping_rule, mob_parameters,
                                            split_var=parent_split.variable_id, pb=pb)
        # update progress bar
        if node.terminal and pb is not None:
            pb.update(node.n_samples)

        return node

    def predict(self, X: np.array) -> np.array:
        """
        MOB target variable predicions for a new array of samples

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        :return: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        # check whether samples to be predicted fall into MOB domain, send a warning otherwise
        check_domain(X, self.tree)

        preds = get_node_preds(self.tree, X)
        return preds

    def save(self, path: str = "./model", force: bool = False, compressed: bool = False) -> None:
        """
        Save the model as pickle object

        :param path: path where to store the pickle object
        :param force: boolean flag, if True overwrite old file with the same path
        :param compressed: boolean flag, if True store a zipped file
        :return: None
        """

        save_pickle_object(self, path, force, compressed)
        return None
