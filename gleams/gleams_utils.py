import altair as alt
import numpy as np
import pandas as pd
from pandas import DataFrame
from category_encoders import TargetEncoder
from plotly import express as px
from scipy.stats import qmc
from numbers import Number

from pymob.mob import MOB
from pymob.classes import RealDomain, RealInterval
from pymob.mob_utils import get_pred_node
from typing import Callable

import warnings


class ParametersError(Exception):
    """Custom Exception Class"""

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def _extract_domain_from_data(array, variable_names):
    """Retrieve domain_dict from the data attribute in Gleams"""
    if variable_names is None:
        range_ = range(array.shape[1])
    else:
        range_ = list(variable_names)

    domain_dict = {id_var: RealInterval(bounds=(np.min(array[:, id_var]), np.max(array[:, id_var])),
                                        included=(True, True)) for id_var, var in enumerate(range_)}

    return domain_dict


def _data_validation(self, data):
    """
    Sanity check on the data argument passed to Gleams.
    In particular, it checks that data is a np.array, pd.DataFrame or dictionary (otherwise raise Exception),
    stores all the relevant information as gleaml attributes: number of variables,
        domains for each variable, if possible variable names

    :param self: the Gleams object
    :param data: the data attribute passed as argument to the Gleams object
    :return: None
    """
    # if dataframe is passed
    if isinstance(data, DataFrame):
        # save variable_names
        self.variable_names = list(data.columns)
        # transform it in array
        data = data.to_numpy()
    # if data is array
    if isinstance(data, np.ndarray):
        # check if data is compliant (does not have any categorical columns)
        if not _is_continuous(data):
            raise Exception("""Categorical Columns have been passed in the data. 
                        Please pass only continuous variables (categories encoded into a single continuous column)
                        and add categorical_columns in the params. If the error persist, try and explicitly cast to
                        int, float, np.int* or np.float* of the columns""")
        data = data.astype(np.float64)
        # save n_features
        self.n_features = data.shape[1]
        # save domain_dict
        self.domain_dict = _extract_domain_from_data(data, self.variable_names)
        # check whether variable_names has been set
        if self.variable_names is None:
            warnings.warn("""Pay Attention: Variable Names not available. 
            If you want explanations with variable names in them, pass it as the "variable_names" parameter""")
            # create variable_names as 1,2, etc
            self.variable_names = [str(id_var) for id_var in self.domain_dict.keys()]

    # if data is the domain dictionary
    elif isinstance(data, dict):
        _validate_domain_dict(data)
        # save domain_dict
        self.domain_dict = data
        #  save n_features
        self.n_features = len(self.domain_dict)
        # override variable_names using the dictionary names
        self.variable_names = list(data.keys())
    # if not one of the previous, Exception
    else:
        raise Exception("You have to pass a pandas DataFrame, a numpy array or a dict!")


def _check_n_sobol_points(n_sobol_points: int) -> None:
    """
    Sanity check on the Gleams attribute 'n_sobol_points'

    :param n_sobol_points: number of Sobol points to be generated, i.e. 2**n_sobol_points
    :return: None
    """
    if not isinstance(n_sobol_points, int):
        warnings.warn("""n_sobol_points should be an integer between 4 and 30,
        the exact number of points to be generated is 2**n_sobol_points.
         Please comply with these boundaries""")
    if n_sobol_points < 4:
        raise Exception("Not enough points to generate")
    if n_sobol_points > 30:
        raise Exception("Too many points to generate")
    return None


def _check_domain(domain_dict, sample):
    """Check if sample fall inside the domain of the node, print a warning if not"""
    domain = RealDomain(domain_dict)
    if not domain.contains(sample):
        warnings.warn(f"WARNING: point {sample} not in domain")


"""utilities"""


def get_sobol_x(n_features: int, n_sobol_points: int, scramble: bool = False, seed: int = 666) -> np.array:
    """
    Generate an array of Sobol points, in the unit sphere [0,1]^{n_features}.
    The array would have size (2**n_sobol_points, n_features)

    :param n_features: number of features on which the black-box model has been trained
        Needed to generate the correct multidimensional size of the Sobol points
    :param n_sobol_points: number of Sobol points to be generated, i.e. 2**n_sobol_points
    :param scramble: boolean flag to obtain different points each time, for more info refer to scipy.stats.qmc docs
    :param seed: seed value, to achieve reproducibility
    :return: array-like, sparse matrix}, shape (2**n_sobol_points, n_features)
        The array of generated sobol points
    """

    # generate the dataset points
    sampler = qmc.Sobol(d=n_features, scramble=scramble, seed=seed)
    sobol_x = sampler.random_base2(m=n_sobol_points)
    return sobol_x


def get_sobol_y(sobol_X: np.array, mode: str, predict_function: Callable) -> np.array:
    """
    Predict the target variable Y for the generated Sobol points, using the prediction function of the black-box model

    :param sobol_X: array-like, sparse matrix, of size (2**n_sobol_points, n_features)
        The array of generated sobol points
    :param mode: one of the following 'regression', 'classification'
        Governs how to obtain predictions from the predict_function
    :param predict_function: prediction function of the black-box model
    :return: array-like, shape (n_samples,)
            The target values (probabilities of class 1 in classification, real numbers in regression).
    """
    sobol_y = predict_function(sobol_X)

    if mode == "classification":
        if len(sobol_y.shape) == 1:
            raise NotImplementedError("In order to run GLIME with binary classification models,"
                                      "please pass the predict_proba method or similar methods "
                                      "that return probability scores")
        elif len(sobol_y.shape) > 2:
            raise NotImplementedError("GLIME currently does not support MultiClass Classification Models")

        else:
            sobol_y = sobol_y[:, 1]
    return sobol_y


def rescale_sobol_points(sobol_X: np.array, domain_dict: dict) -> np.array:
    """
    Rescale sobol array of the X variables, to have the same domain of the original data

    :param sobol_X: {array-like, sparse matrix}, shape (2**n_sobol_points, n_features)
        The array of generated sobol points
    :param domain_dict: dictionary of the full Gleams domain
    :return: {array-like, sparse matrix}, shape (2**n_sobol_points, n_features)
        Array of the generated Sobol points, rescaled on the Gleams domain
    """
    for id_var, (var, bounds) in enumerate(domain_dict.items()):
        min_var, max_var = bounds

        # rescale sobol points in the range of the original variables
        sobol_X[:, id_var] = sobol_X[:, id_var] * (max_var - min_var) + min_var

    return sobol_X


def get_model(X, y, minsplit, mob_method, aggregation_function, verbose, stopping_value, quantile, ml_continuous,
              max_outliers):
    """Instantiate and fit a MOB model with the specified parameters. Return the fitted model"""
    mob = MOB(minsplit=minsplit, method=mob_method, aggregation_function=aggregation_function, verbose=verbose,
              stopping_value=stopping_value, quantile=quantile, ml_continuous=ml_continuous, max_outliers=max_outliers)
    mob.fit(X, y)

    return mob


def get_coefficients(mob: MOB, sample: np.array, standardized: bool = False) -> tuple:
    """
    Retrieve intercept and coefficients of the Local Linear Model of the leaf where 'sample' lies.
    It is possible to obtain both unstandarized and standardized coefficients.

    :param mob: fitted object of the MOB class
    :param sample: array-like, sparse matrix, of size (n_features,)
        The datapoint for which we want local coefficients
    :param standardized: boolean flag, if True local coefficients are standardized
    :return: a tuple containing the dictionary of coefficients, e.g. {"name_var1": coeff1}, and the intercept (float)
    """

    node = get_pred_node(mob.tree, sample.ravel())
    coef = node.regression.coef_
    intercept = node.regression.intercept_
    if standardized:
        std_x, std_y = node.st_devs
        try:
            coef = np.round(np.multiply(coef, std_x) / std_y, 10)
        except:
            coef = np.zeros(coef.shape)
    return coef, intercept


def create_explanation_dictionary(coefficients: np.array, intercept: float, variable_names: list) -> dict:
    """
    Create and return a dictionary containing info about the given explanation,
     i.e. each variable linked to its coefficient value, intercept value and ranking of importance among variables
    """
    # sort the coefficient in decreasing absolute order (the highest coefficients on top)
    ids = np.argsort(np.abs(coefficients))[::-1]
    # create a dictionary with {name_var : coefficient} in decreasing abs order
    expl_dict = dict(zip(np.array(variable_names), coefficients))
    expl_dict_sorted = dict(zip(np.array(variable_names)[ids], coefficients[ids]))

    return {"coefficients": expl_dict, "sorted_coefficients": expl_dict_sorted, "intercept": intercept,
            "sorting_ids": ids}


def make_bars_coef(coef, variable_names, title=None):
    """Create the barplot with each coefficient represented as a (positive or negative) bar,
     with bar length based on its magnitude"""

    # group info in bars DataFrame
    bars = pd.DataFrame(index=list(map(str, variable_names)))
    bars["coef"] = coef
    bars['positive'] = bars["coef"] > 0

    # inspired by: https://plotly.com/python-api-reference/generated/plotly.express.bar.html
    bars['positive'] = bars['positive'].map({True: 'positive', False: 'negative'})
    bars["abs_coef"] = np.absolute(bars["coef"])
    # sort df wrt abs_coeff
    bars = bars.sort_values(by="abs_coef", axis=0, ascending=False, ignore_index=False)

    # create barplot
    fig_px = px.bar(data_frame=bars, x="coef", y=bars.index.values.tolist(), labels={"y": "variables"},
                    color="positive", title=title,
                    color_discrete_map={
                        "negative": 'red',
                        "positive": 'green'
                    }).update_yaxes(categoryorder="total ascending")

    return fig_px


# check continuous values (no casting is applied)
def _is_continuous(matrix):
    is_numeric = np.vectorize(lambda val: isinstance(val, Number), otypes=[bool])
    return all(is_numeric(matrix.ravel()))


def _validate_domain_dict(domain_dict):
    """
    Sanity check on the attribute domain_dict of Gleams:
        control if the max and min boundaries are numbers
        Control if max is striclty greater than min
        (if equal there will be problems in standardizing the coefficients and in Linear Regression)
        Control if there are ids as keys of the dictionary (not variable names)
    """
    for id_var, bounds in domain_dict.items():
        min, max = bounds
        if not (isinstance(max, Number) and isinstance(min, Number)):
            raise Exception("The domain dictionary must contain only numeric variables."
                            "Please convert the categorical variables into numeric "
                            "and pass the encoded dataset/dictionary domain")
        if not max > min:
            raise Exception(
                "The range of the variable {} is wrong (max boundary > min boundary or they are the same value".format(
                    id_var))

def vconcat(list):
    """Concatenate the list of figures in a single altair figure"""
    return alt.vconcat(*list).resolve_scale(color='independent')