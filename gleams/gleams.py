"""
Gleams Explanation Model
"""

import numpy as np
from pymob.mob_utils import get_leaf_weights, get_leaves
from sklearn.utils.validation import check_array

from gleams_utils import (ParametersError,
                          _check_domain, _check_n_sobol_points,
                          _data_validation,
                          get_coefficients, get_model, get_sobol_x,
                          get_sobol_y, rescale_sobol_points, create_explanation_dictionary, make_bars_coef)

from utils import save_pickle_object


class Gleams:

    def __init__(self, data: np.array, n_sobol_points: int, predict_function: object, mode: str = "regression",
                 minsplit: int = 50, mob_method: str = 'MSE fluc process', aggregation_function: str = "maxsum",
                 verbose: bool = False, stopping_value: float = None, quantile: float = None,
                 ml_continuous: bool = True, max_outliers: int = 5, variable_names: list = None,
                 categorical_features_dict: dict = dict(), is_keras: bool = False) -> None:
        """
        Gleams explanation class.
        The class provides an explanation model, equipped with the fit, predict methods
        and various methods to obtain different types of explanations.

        Gleams internally exploits the MOB model to recursively partition the input space
        and compute a linear piecewise surrogate model approximating the black-box model.
        Such model provides different types of global and local explanations.

        :param data: {array-like, sparse matrix}, shape (n_samples, n_features),
            accepts also pd.DataFrame or a dictionary of shape { id_var : [min_bound, max_bound]}
            The data samples used to train the black-box model.
        :param n_sobol_points: number of Sobol points to be generated, i.e. 2**n_sobol_points
        :param predict_function: the predict function of the black-box model to be explained
        :param mode: one of the following: 'regression', 'classification'.
        :param minsplit: minimum number of points in each terminal node of the MOB model
        :param mob_method: one of the following 'MSE fluc process','naive R2','online R2','weighted online R2',
            'loglik fluc process'. Defines the MOB splitting criterion.
        :param aggregation_function: one of the following 'maxmax', 'maxsum'.
            Specifies how to aggregate a multidimensional process in the best split quest.
        :param verbose: boolean flag to print Gleams info
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
        :param variable_names: list of variable names, in the same order as the 'data' columns
        :param categorical_features_dict: dictionary of the shape {name_categorical_variable : fitted Encoder}
            Should contain the Encoder fitted on each categorical variable prior to training the black-box model
        :param is_keras: boolean flag, if True the black-box model is a keras model
        """

        self.mode = mode
        self.predict_function = predict_function
        self.minsplit = minsplit
        self.max_outliers = max_outliers
        self.stopping_value = stopping_value
        self.verbose = verbose
        self.variable_names = variable_names
        self.ml_continuous = ml_continuous
        self.is_keras = is_keras
        self.mob_method = mob_method
        self.aggregation_function = aggregation_function
        self.quantile = quantile

        # validate the data atribute and extract relevant info: self.variable_names, self.n_features, self.domain_dict
        self.domain_dict = None
        _data_validation(self, data)

        self.mob = None
        self.is_fitted_ = False

        _check_n_sobol_points(n_sobol_points)
        self.n_sobol_points = n_sobol_points

    def fit(self) -> object:
        """
        Fit the Piecewise Linear surrogate model, i.e. the explanation model.

        :return: the trained object
        """

        X = get_sobol_x(self.n_features, self.n_sobol_points, scramble=False, seed=666)
        X = rescale_sobol_points(X, self.domain_dict)
        y = get_sobol_y(X, self.mode, self.predict_function)

        self.mob = get_model(X=X, y=y, minsplit=self.minsplit, mob_method=self.mob_method,
                             aggregation_function=self.aggregation_function, verbose=self.verbose,
                             stopping_value=self.stopping_value, quantile=self.quantile,
                             ml_continuous=self.ml_continuous,
                             max_outliers=self.max_outliers)
        self.is_fitted_ = True

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Gleams predicted target values for the X array of samples,
            i.e. predictions of the piecewise linear surrogate model (which closely emulate black-box model predictions)
        Gleams predictions have the advantage of being computed usually much faster than using the black-box model,
            since the piecewise model is relative simple and fast to query.
        Predictions are as accurate as the R2 score achieved by Gleams (basically the gleams 'stopping_value' attribute)
        Internally, Gleams calls the trained MOB predict function.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features), accepts also pd.DataFrame
            The samples to be predicted.
        :return: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        """

        _check_domain(self.domain_dict, X)
        return self.mob.predict(X)

    def check_performance(self):
        """Print Goodness of fit information on the Gleams surrogate model"""

        n_sobol_points = 2 ** self.n_sobol_points
        leaves = get_leaves(self.mob.tree)
        num_leaves = len(leaves)
        r2 = sum([leaf.score * leaf.n_samples / n_sobol_points for leaf in leaves])
        avg_leaf_size = n_sobol_points / num_leaves
        print("Gleams Goodness of Fit info:")
        print(f"{n_sobol_points=}, {num_leaves=:.2f}, {r2=:.2f}, {avg_leaf_size=:.2f}")

    def save(self, path: str = "./global_exp", force: bool = False, compressed: bool = False) -> None:
        """
        Save the global explanation model as pickle object.

        :param path: path where to store the pickle object
        :param force: boolean flag, if True overwrite old file with the same path
        :param compressed: boolean flag, if True store a zipped file
        :return: None
        """

        save_pickle_object(self, path, force, compressed)
        return None

    def local_importance(self, sample: np.array, visualization_mode: str = "notebook", standardized: bool = True,
                         show: bool = True,
                         save: bool = False) -> tuple:
        """
        Compute local importance for a specific sample,
         i.e. coefficients of the local linear model in the region where 'sample' lies.
         Provides also the local importance plot, i.e. a bar plot of the coefficients, ordered by magnitude

        :param sample: array of shape (n_features,).
            Individual for which we request the local explanation,
            can be a row of the train or test dataset,or a new individual.
        :param visualization_mode: plotly visualization mode
        :param standardized: boolean flag, if True local coefficients are standardized
        :param show:  boolean flag, if True the local importance plot is shown
        :param save:  boolean flag, if True save the local importance plot on disk
        :return: a tuple containing the explanation dictionary and the local importance plot
        """

        # check if sample lies in the Gleams training input space
        _check_domain(self.domain_dict, sample)
        coefficients, intercept = get_coefficients(self.mob, sample, standardized=standardized)
        explanation_dictionary = create_explanation_dictionary(coefficients, intercept, self.variable_names)

        fig = make_bars_coef(coefficients, self.variable_names)
        if show:
            fig.show(renderer=visualization_mode)
        if save:
            fig.write_image("local_importance.pdf")

        return explanation_dictionary, fig

    def global_importance(self, true_to="model", data=None, meaning="ranking importance", visualization_mode="notebook",
                          show=True, plot_title=None, save=False, path="./global_importance_true_to_model.pdf"):
        """
        Compute global importance on the entire input space of the black-box model,
        i.e. average the coefficients of each local linear model,
        using different rationales to provide different interpretations.

        In particular, it is possible to obtain:
        i) an unbiased ranking of the features based on their importance on the black-box model prediction,
        ii) the average impact of each feature on the model predictions
            (to understand whether the feature has on average a positive or negative impact),
        The two rationales above are available in:
        i) 'true to the data': focus only on the input space regions where real-world data lies
        ii) 'true to the model': consider the entire input space to understand
            how the variables have an impact on the model in general.
        To know more about the 'true to the data', 'true to the model' distinction, refer to:
        "True to the Model or True to the Data?", Chen et al. 2020. paper link: https://arxiv.org/pdf/2006.16234.pdf

        Provides also the global importance plot, i.e. a bar plot of averaged local coefficients, ordered by magnitude

        :param true_to: one of the following 'data', 'model'
    :   param data: {array-like, sparse matrix}, shape (n_samples, n_features), accepts also pd.DataFrame
            The data samples on which to compute importance of the local leaves. Needed only if true_to='data'
        :param meaning:  one of the following 'ranking importance', 'average impact'
        :param visualization_mode: plotly visualization mode
        :param show:  boolean flag, if True the local importance plot is shown
        :param plot_title: plot string to be used in the figure
        :param save:  boolean flag, if True save the local importance plot on disk
        :param path: path of the file where to store the global importance figure (only if save=True)
        :return: a tuple containing the explanation dictionary and the global importance plot
        """

        if true_to == "model":
            global_coefficients = np.zeros((len(self.mob.leaves), len(self.variable_names)))
            global_intercept = 0
            weights = np.zeros((len(self.mob.leaves),))
            n_sobol_points = self.mob.tree.n_samples
            # iterate on each leaf, to obtain the coeffs average
            for id_leaf, leaf in enumerate(self.mob.leaves):
                global_intercept += leaf.regression.intercept_
                # rescale coefficients by the local standard deviations (inside the leaf) of the Sobol data
                try:
                    global_coefficients[id_leaf] = np.multiply(leaf.regression.coef_, leaf.st_devs[0]) / leaf.st_devs[1]
                except:
                    global_coefficients[id_leaf] = np.zeros((len(self.variable_names),))
                weights[id_leaf] = leaf.n_samples / n_sobol_points
            # aggregate the coeffs importance
            if meaning == "ranking importance":
                global_coefficients = np.abs(global_coefficients)
            elif meaning == "average impact":
                pass
            else:
                raise Exception(f"Inexistent 'meaning' parameter: {meaning}")
            global_coefs = np.average(global_coefficients, axis=0, weights=weights)

            if plot_title is None:
                plot_title = "True to the Model, Global Feature Importance"

        elif true_to == "data":

            try:
                data = check_array(data)
            except Exception:
                print("""No valid dataset passed to the global_importance method. 
                When true_to="data", a valid dataset is mandatory to compute the weight of each leaf.""")

            leaf_weights = get_leaf_weights(self.mob.tree, data)
            leaf_coefs = {leaf.id: leaf.regression.coef_ for leaf in self.mob.leaves}
            leaf_stdevs = {leaf.id: leaf.st_devs for leaf in self.mob.leaves}
            intercept = [leaf.regression.intercept_ for leaf in self.mob.leaves]
            global_intercept = np.sum(intercept)

            global_coefficients = np.zeros((len(leaf_weights), len(self.variable_names)))
            for row_id, leaf_id in enumerate(leaf_weights.keys()):
                # standardize coefficients by local standard dev
                try:
                    global_coefficients[row_id] = np.multiply(leaf_coefs[leaf_id], leaf_stdevs[leaf_id][0]) / \
                                                  leaf_stdevs[leaf_id][1]
                except:
                    global_coefficients[row_id] = np.zeros((len(self.variable_names),))
            # aggregate the coeffs importance
            if meaning == "ranking importance":
                global_coefficients = np.abs(global_coefficients)
            elif meaning == "average impact":
                pass
            else:
                raise Exception(f"Inexistent 'meaning' parameter: {meaning}")

            weights = [leaf_weights[key] for key in sorted(leaf_weights.keys(), reverse=False)]

            global_coefs = np.average(global_coefficients, axis=0, weights=weights)

            if plot_title is None:
                plot_title = "True to the Data, Global Feature Importance"

        else:
            raise ParametersError(message="Invalid value for the true_to parameter")

        fig = make_bars_coef(global_coefs, self.variable_names, title=plot_title)
        if show:
            fig.show(renderer=visualization_mode)
        if save:
            fig.write_image(path)

        explanation_dictionary = create_explanation_dictionary(global_coefs, global_intercept, self.variable_names)
        return explanation_dictionary, fig


    def get_singlevar_whatif_nodes(self, sample: np.array, var_id: int) -> list:
        """
        Returns all the MOB leaves containing the what-if sample on the selected variable, i.e.
            datapoints with the selected variable free to span any value,
            while the other variables are kept fixed to the value they have in 'sample'.

        :param sample: array of shape (n_features,).
            Individual for which we request the single variable what-if explanation,
            can be a row of the train or test dataset,or a new individual.
        :param var_id: id of the what-if variable in the 'sample' array
        :return: list of nodes traversed by the what-if 'sample' changing only the selected variable
        """

        global_domain = self.mob.domain_dict
        node_list = [node for node in self.mob.leaves if node.domain.contains_without_var(sample, var_id,
                                                                                          global_domain=global_domain)]
        return node_list

    def whatif_global_importance(self, sample: np.array, standardize: str = "global") -> np.array:
        """
        Compute the what-if importance for each variable,
            by giving it freedom to change along the entire MOB inout space,
            while keeping the other variables fixed to 'sample' value.
        The importance of the variable consists in the average of the standardized coefficients
        of all the nodes traversed during the what-if analysis

        :param sample: array of shape (n_features,).
            Individual for which we request the single variable what-if explanation,
            can be a row of the train or test dataset,or a new individual.
        :param standardize: one of the following 'global', 'local', None
            If 'global', the local coefficients are standardized against
                the variable standard deviation  on the entire MOB input space,
            if 'local', the standardization is done through the variable standard deviation on the specific leaf,
            if None, the unstandardized coefficients will be used.
        :return: the averaged standardized coefficients across all the what-if nodes
        """

        try:
            global_std_x, global_std_y = self.mob.global_st_devs

            # initialize the averaged what-if coefficients
            what_if_coeffs = np.zeros(sample.shape[0])

            for var_id in range(sample.shape[0]):
                # retrieve waht-if nodes, coefficients and node weights (hyper-volume)
                nodes_list = self.get_singlevar_whatif_nodes(sample, var_id)
                leaves_coeffs = np.zeros(len(nodes_list))
                weights = np.zeros(len(nodes_list))

                for coeffs_id, node in enumerate(nodes_list):
                    min_local_boundary, max_local_boundary = node.domain[var_id]
                    weights[coeffs_id] = max_local_boundary - min_local_boundary
                    coef = node.regression.coef_[var_id]
                    local_std_x, local_std_y = node.st_devs
                    # standardize coefficients
                    if standardize == "global":
                        std_coef = coef * global_std_x[var_id] / global_std_y
                    elif standardize == "local":
                        std_coef = coef * local_std_x[var_id] / global_std_y
                    elif standardize == None:
                        std_coef = coef
                    else:
                        raise Exception(
                            "Non valid 'standardize' parameter. It should be one of: 'global', 'local', None")
                    leaves_coeffs[coeffs_id] = std_coef
                # average coefficients (weighted by the hyper-volume or unweighted)
                if standardize == "local":
                    what_if_coeffs[var_id] = np.sum(np.abs(leaves_coeffs), axis=0)
                else:
                    what_if_coeffs[var_id] = np.average(np.abs(leaves_coeffs), axis=0, weights=weights)
        except:
            raise Exception("Untracked Exception")
        return what_if_coeffs
