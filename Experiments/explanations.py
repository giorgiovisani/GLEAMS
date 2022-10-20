import os
import time
import warnings

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from shap import DeepExplainer, KernelExplainer, TreeExplainer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import partial_dependence
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from gleams.gleams import Gleams
from utils import load_pickle_object


class SKlearnModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Class that takes as input a generic Regressor model,
    wraps it into a Sklearn-like model, providing the fit and predict methods.

    Useful to use Sklearn functions working only on sklearn Regressors on generic Regressors
    (in this script this will be used for the partial_dependence function)
    """

    def __init__(self, original_model, **kwargs):
        self.original_model = original_model
        self._estimator_type = "regressor"
        self.kwargs = kwargs

    def fit(self, X, y):
        # necessary code to run a valid slearn estimator
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        # necessary code to run a valid slearn estimator
        check_is_fitted(self)
        X = check_array(X)
        # my code
        predictions = self.original_model.predict(X, self.kwargs)
        return predictions


def train_gleams(n_sobol_points, model, X_data, is_keras):
    """Train the Gleams explanation method, with default parameters.
        The only modifiable parameters are n_solo_points, the blackbox model to be used and the data array"""

    glob_exp = Gleams(data=X_data, n_sobol_points=n_sobol_points, predict_function=model.predict, mode="regression",
                      minsplit=20, mob_method='MSE fluc process', aggregation_function="maxsum", verbose=False,
                      stopping_value=0.95, quantile=None, ml_continuous=True, max_outliers=5,
                      variable_names=list(X_data.columns), is_keras=is_keras)

    init_time = time.time()
    glob_exp.fit()
    end_time = time.time()
    train_time = end_time - init_time
    print(f"{train_time=:.2f}")

    return glob_exp


def train_shap(model, X_data, model_type, irrelevant_regr=False):
    """Train Shap explanations on the given model and data.
        Depending on the model type, different Explainers will be used.
        In particular: if xgboost model --> TreeExplainer,
                       if neural network --> DeepExplainer
                       if IrrelevantVarsRegressor --> KernelExplainer

        The choice of Explainers is determined by the computational speed,
        KernelExplainer is extremly slow, but works on any tipe of model,
        therefore we use it only for IrrelevantVarsRegressor model (which is not considered of any known type by Shap),
        For Tree models and Neural Networks, custom faster implementations are chosen
        """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if model_type == "xgb" and not irrelevant_regr:
            return train_shap_xgb(model, X_data)
        elif model_type == "nn" and not irrelevant_regr:
            return train_shap_nn(model, X_data)
        elif irrelevant_regr:
            return train_shap_irrelevant(model, X_data)
        else:
            raise Exception("Invalid 'model' parameter")


def train_shap_irrelevant(model, X_data):
    """Train Shap on IrrelevantVarsRegressor, using KernelExplainer"""

    explainer = KernelExplainer(model.predict, data=X_data, link="identity", verbose=0)
    # compute SHAP attributions using only 500 variables' combinations,
    # due to computational reasons (too long to compute). For the rebuttal we will compute the whole attributions
    shap_values = explainer.shap_values(X_data, nsamples=500)
    return shap_values


def train_shap_nn(model, X_data):
    """Train Shap on Neural Network, using DeepExplainer"""

    explainer = DeepExplainer(model=model, data=X_data.values)
    shap_values = explainer.shap_values(X_data.values)[0]
    return shap_values


def train_shap_xgb(model, X_data):
    """Train Shap on Xgboost, using TreeExplainer"""

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    return shap_values


def get_lime(train_data, cat_indices):
    """Instantiate a LimeTabularExplainer object, to produce explanations
        on any of the black-box models used in our experiments"""

    colnames = list(train_data.columns)
    explainer = LimeTabularExplainer(train_data.values,
                                     feature_names=colnames,
                                     mode='regression',
                                     verbose=False,
                                     kernel_width=None,
                                     discretize_continuous=True,
                                     categorical_features=cat_indices)
    return explainer


def train_lime(explainer, sample, predict_function, num_features):
    """Train the already instantiated LimeTabularExplainer object,
        on the given model (predict_function passed as argument).

        Returns LIME attributions for a given 'sample', i.e. the coefficients of the local linear model
        """

    exp = explainer.explain_instance(sample, predict_function, num_samples=5000, num_features=num_features)

    raw_attributions = exp.local_exp[0]
    raw_attributions.sort(reverse=False)
    lime_attributions = [attr for id_var, attr in raw_attributions]
    return lime_attributions


def train_pdp(model, data):
    """Train the partial dependence plots explanations (using partial_dependence function in sklearn)
        on a Sklearn-like model obtained wrapping our model into the SKlearnModelWrapper.

        Return PDP feature attributions, exploiting the Greenwell idea of computing the variance of PDP functions"""

    # sklearn wrapper to use the partial_dependence function
    sklearn_wrapper = SKlearnModelWrapper(model)
    sklearn_wrapper.fit(data, np.ones((data.shape[0],)))

    feature_attributions = list()
    for id_var in list(range(data.shape[1])):
        try:
            explainer = partial_dependence(estimator=sklearn_wrapper, X=data,
                                           features=[id_var],
                                           kind="average",
                                           grid_resolution=100,
                                           percentiles=(0.05, 0.95)
                                           )
            pdp_values = explainer["average"]
            # variance of pdp_values,inspired by: Greenwell paper: https://arxiv.org/pdf/1805.04755.pdf
            # Greenwell, Simple and Effective Model-Based Variable Importance Measure, 2018.
            feature_attribution = np.var(pdp_values)
            feature_attributions.append(feature_attribution)
        except:
            feature_attributions.append(0)

    return feature_attributions


def load_previous_expl(x_methods, custom_explanation_path_function):
    """Load stored feature attributions obtained from various explanation methods.
        Return a dictionary of attributions for each considered explanation method"""

    attributions = {x_method: None for x_method in x_methods}

    for x_method in x_methods:
        if os.path.exists(custom_explanation_path_function(x_method)):
            attributions[x_method] = load_pickle_object(custom_explanation_path_function(x_method))
        else:
            pass
    return attributions
