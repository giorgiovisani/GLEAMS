import pandas as pd
import numpy as np
import time

from lime.lime_tabular import LimeTabularExplainer
import shap
from shap import TreeExplainer, DeepExplainer, KernelExplainer
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
from functools import partial

from gleams.gleams import Gleams


def custom_predict(predict_function):

    custom_predict_function = partial(predict_function,verbose=0)
    return custom_predict_function

class SKlearnModelWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, original_model='xgb',**kwargs):
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


def check_transform_data_array(data):
    colnames = None
    if isinstance(data, pd.DataFrame):
        colnames = list(data.columns)
        data = data.values
    if not isinstance(data, np.ndarray):
        raise Exception("Data passed is not DataFrame nor Numpy Array")
    return data, colnames

def train_gleams(n_sobol_points, model, X_data, is_keras):


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if model_type == "xgb" and not irrelevant_regr:
            return train_shap_xgb(model,X_data)
        elif model_type == "nn" and not irrelevant_regr:
            return train_shap_nn(model,X_data)
        elif irrelevant_regr:
            return train_shap_irrelevant(model, X_data)
        else:
            raise Exception("Invalid 'model' parameter")


def train_shap_irrelevant(model, X_data):
    # explainer = SamplingExplainer(model=model,data=X_data)
    # explainer = Explainer(model=model.predict,data=X_data)

    explainer = KernelExplainer(model.predict,data=X_data,link="identity", verbose=0)
    # TODO: in teoria servirebbero piu samples (il default sarebbe 2*n_features+2048), dovrei anche usare tutti i dati di test (invece adesso sto facendo subsampling con .sample)
    # To use less data in shap: shap.sample(X_data,200) in shap_values
    shap_values = explainer.shap_values(shap.sample(X_data,500), nsamples=500)
    return shap_values

def train_shap_nn(model, X_data):
    # explainer = KernelExplainer(model.predict,data=X_data,link="identity",method="deep")
    explainer = DeepExplainer(model=model,data=X_data.values)
    shap_values = explainer.shap_values(X_data.values)[0]
    return shap_values

def train_shap_xgb(model, X_data):
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    return shap_values



def get_lime(train_data, cat_indices):
    # train_data,colnames = check_transform_data_array(train_data)
    colnames = list(train_data.columns)
    explainer = LimeTabularExplainer(train_data.values,
                                     feature_names=colnames,
                                     mode='regression',
                                     verbose=False,
                                     kernel_width=None,
                                     discretize_continuous=True,
                                     categorical_features=cat_indices)
    return explainer

def train_lime(explainer, test_unit, predict_function, num_features):
    exp = explainer.explain_instance(test_unit, predict_function, num_samples=5000, num_features=num_features)

    raw_attributions = exp.local_exp[0]
    raw_attributions.sort(reverse=False)
    lime_attributions =[attr for id_var,attr in raw_attributions]
    return lime_attributions


def train_pdp(model, data, model_type="xgb"):

    # sklearn wrapper to use the partial_dependence function
    sklearn_wrapper = SKlearnModelWrapper(model)
    sklearn_wrapper.fit(data,np.ones((data.shape[0],)))

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
            # variance of pdp_values,
            # inspired from Greenwell, Simple and Effective Model-Based Variable Importance Measure, 2018
            # paper link: https://arxiv.org/pdf/1805.04755.pdf
            feature_attribution = np.var(pdp_values)
            feature_attributions.append(feature_attribution)
        except:
            feature_attributions.append(0)

    return feature_attributions



