"""
Evaluating different explanation methods using monotonicity and recall of important features
"""


import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm

import pandas as pd
from scipy.stats import spearmanr
import xgboost
from numpy.matlib import repmat

from sklearn.model_selection import train_test_split
import pymob.classes as classes
from ml_models import _check_params, import_preproc_dataset, load_nn, save_pkl_model, save_keras_zip_model, get_xgb_regr, get_nn_regr, EarlyStopping_verbose

from explanations import train_gleams, train_pdp, train_lime, get_lime, train_shap
from pymob.mob_utils import check_domain, get_most_crowded_leaf

from utils import load_pickle_object, save_pickle_object

np.seterr(all='raise')


# wrapper to be able to use the model on all the coordinates

class IrrelevantVarsRegressor:
    """Model of the given model class, trained with some features that are irrelevant for the model"""

    def __init__(self, model_class=None, model=None, **kwargs):
        self.kwargs = kwargs
        self.trained = False
        self._define_model(model_class, model)

    def _check_irrelevant_vars_params(self, irrelevant_vars_ids, n_irrelevant_ids):
        if all([not irrelevant_vars_ids, not n_irrelevant_ids]):
            raise Exception("You must pass one (and only one) of 'irrelevant_vars_ids' or 'n_ids_to_extract'")
        if all([irrelevant_vars_ids, n_irrelevant_ids]):
            raise Exception("You must pass only one of 'irrelevant_vars_ids' or 'n_ids_to_extract'")
        if all([irrelevant_vars_ids, not isinstance(irrelevant_vars_ids, list)]):
            raise Exception("'irrelevant_vars_ids' must be a list!")

    def _define_model(self, model_class, model):
        if model:
            self._restricted_model = model
            self.model_class = type(model)

        elif model_class:
            self._restricted_model = model_class(**self.kwargs)
            self.model_class = model_class
        else:
            raise Exception("Both 'model' and 'model_class' params are empty."
                            "You must pass a valid model instance or a model class with optional kwargs params")

    def get_random_irrelevant_ids(self, n_irrelevant_ids, max_id):
        """Get a list of random ids of length n_ids_to_extract,
            ids range from 0 to max_id"""
        #Fixed the seed so that we obtain always the same relevant variables ids
        random.seed(42)
        return random.sample(range(max_id), n_irrelevant_ids)

    def _restrict_data(self, data):
        """Returns a restricted dataset containing only the variables considered relevant"""
        is_df = False

        if isinstance(data, pd.DataFrame):
            is_df = True
            colnames = list(data.columns)
            data = data.values

        if isinstance(data, np.ndarray):
            restricted_data = np.delete(data, self.irrelevant_vars_ids, axis=1)
            if is_df:
                restricted_colnames = [var_name for id_var, var_name in enumerate(colnames) if
                                       id_var not in self.irrelevant_vars_ids]
                restricted_data = pd.DataFrame(restricted_data, columns=restricted_colnames)
            return restricted_data
        else:
            raise Exception("Data passed is not DataFrame nor Numpy Array")

    def _restrict_validation_data(self, validation_list):
        """Dynamically restrict validation data inside the fit function (if validation data has been passed as kwargs)"""
        X_val, y_val = validation_list[0]
        restricted_X_val = self._restrict_data(X_val)
        if self.model_class == xgboost.sklearn.XGBRegressor:
            validation_data = [(restricted_X_val, y_val)]
        else:
            validation_data = (restricted_X_val, y_val)
        return validation_data

    def fit(self, train_data, train_labels, irrelevant_vars_ids=None, n_irrelevant_ids=None, **kwargs):

        # get the irrelevant variables ids
        self._check_irrelevant_vars_params(irrelevant_vars_ids=irrelevant_vars_ids, n_irrelevant_ids=n_irrelevant_ids)
        self.irrelevant_vars_ids = irrelevant_vars_ids if irrelevant_vars_ids else self.get_random_irrelevant_ids(
            n_irrelevant_ids=n_irrelevant_ids, max_id=train_data.shape[1])

        # restrict the data to the meaningful variables only
        restricted_train_data = self._restrict_data(train_data)

        # restrict validation data if passed as kwargs
        if kwargs.get("eval_set", None):
            kwargs["eval_set"] = self._restrict_validation_data(kwargs["eval_set"])
        if kwargs.get("validation_data", None):
            kwargs["validation_data"] = self._restrict_validation_data(kwargs["validation_data"])

        # train a restricted model on the restricted_train_data
        self._restricted_model.fit(restricted_train_data, train_labels, **kwargs)
        self.trained = True
        return None

    def predict(self, test_data):
        if not self.trained:
            raise Exception("You must train the model before using predict")

        restricted_test_data = self._restrict_data(test_data)
        kwargs = {"verbose": 0} if self.model_class == "keras.engine.sequential.Sequential" else dict()

        return self._restricted_model.predict(restricted_test_data, **kwargs)

def get_lime_cat_indices(dataset):
    if dataset == "wine":
        cat_ids = None
    elif dataset == "parkinson":
        cat_ids=[1,]
    elif dataset == "houses":
        cat_ids = [0,1,4,5,6,7,8,12,18]
    return cat_ids

def get_perturbed_examples(original_example, var_id, mc_repetitions, mc_boundaries):
    """
    Generate array of perturbed examples, exactly the same as the original example
    apart from the variable var_id which exhibits perturbed values in the given boundary

    :param original_example:
    :param var_id:
    :param mc_repetitions:
    :return:
    """

    if not isinstance(mc_boundaries, classes.RealInterval):
        raise Exception("'mc_boundaries' must be a RealInterval object")

    # simulate data for the given coordinate
    min, max = mc_boundaries
    perturbed_variable_values = np.random.uniform(min, max, size=(mc_repetitions,))
    perturbed_examples = repmat(original_example, mc_repetitions, 1)
    perturbed_examples[:, var_id] = perturbed_variable_values
    return perturbed_examples


def compute_single_monoton(e_list, a_list):
    """
    Computing the monotonicity metric from Nguyen and Martinez paper (https://arxiv.org/pdf/2007.07584.pdf):
     Spearman's correlation between the absolute values of attributions and expected loss
    when removing features (Eq. (1) of the paper).

    Creates a model with certain variables not used (which should have zero attribution in the explanation)
    Technically, we compute a smaller model using the variables deemed meaningful.
    Then we generate a wrapper around the smaller model, which takes as input the entire set of variables.
    In this way, we obtain a model with some variables which have no impact on prediction

    
    INPUT:
        example: the example at hand
        model: the model at hand
        a_array: vector with absolute values of attribution
        n_mc: number of points for the Monte-Carlo simulations
        
    OUTPUT:
        spearman(a_array,e_array)
    """
    a_list = np.abs(a_list)
    corr, _ = spearmanr(a_list, e_list)
    # e_list = [round(val,2) for val in e_list]
    # a_list = [round(val,2) for val in a_list]
    # print(f"{e_list=}")
    # print(f"{a_list=}")
    # print(f"{corr=}")
    return corr


def compute_single_example_e_list(example, model, mc_boundaries, n_mc=1000, is_keras=False):
    """
    The function takes each example separately,
        For each variable, computes n_mc random values of the variable (inside the passed boundary)
        and generates new examples differing from the original one only in the given variable value.
        Using the ML model, predicts the original example and the perturbed examples,
        and compute the mean of the square difference between the original and perturbed examples ML predictions.


    Creates a model with certain variables not used (which should have zero attribution in the explanation)
    Technically, we compute a smaller model using the variables deemed meaningful.
    Then we generate a wrapper around the smaller model, which takes as input the entire set of variables.
    In this way, we obtain a model with some variables which have no impact on prediction


    INPUT:
        example: the example at hand
        model: the model at hand
        a_array: vector with absolute values of attribution
        n_mc: number of points for the Monte-Carlo simulations

    OUTPUT:
        spearman(a_array,e_array)

    """

    dim = example.shape[0]
    e_list = [0] * dim

    predict_kwargs = {"verbose": 0} if is_keras==True else {}
    # look at all features
    for i in range(dim):
        variable_boundaries = mc_boundaries[i]

        # get perturbed predictions using the irrelevant model
        perturbed_examples = get_perturbed_examples(original_example=example, var_id=i, mc_repetitions=n_mc,
                                                    mc_boundaries=variable_boundaries)
        y_pert = model.predict(perturbed_examples, **predict_kwargs)

        # get the prediction for the true example, using the model
        y_pred = model.predict(example.reshape(1, -1), **predict_kwargs)[0]

        # Monte-Carlo extimate of the L2 loss between the two
        e_list[i] = np.mean(np.square(y_pred - y_pert))

    return e_list


def compute_recall(true_attributions, explanation_attributions, n_irrelevant_vars=5):
    """
    Recall of important features, following Ribeiro et al. (2016). Given a list
    of important features and the ordered feature attribution vector given by the explanation,
    look at how many of the top K features of a are actually important.
    
    INPUT
        true_features: list of important features
        a_array: feature attribution vector absolute values
        K: number of features to consider (Ribeiro takes K=10)
        
    OUTPUT
        recall of important features
    """
    n_relevant_vars = len(true_attributions) - n_irrelevant_vars

    sorted_expl_attributions = sorted(enumerate(np.abs(explanation_attributions)), key=lambda x: x[1], reverse=True)
    estimated_relevant_features = sorted_expl_attributions[:n_relevant_vars]
    estimated_relevant_features_ids = [id_var for id_var, attr in estimated_relevant_features]

    true_relevant_features_ids = [id_var for id_var, attr in enumerate(true_attributions) if attr]

    correct_relevant_features = [id_var for id_var in estimated_relevant_features_ids if
                                 id_var in true_relevant_features_ids]
    recall = len(correct_relevant_features) / n_relevant_vars

    return recall


def explanation_path(x_method, dataset, model, n_sobol_points, irrelevant=False, relevant_vars=None,local=None):
    """Get the Explanation Path where to store computed explanation or to load an already existing explanation"""

    explanation_folder = "./explanations"
    params_in_name = f"sobol{n_sobol_points}" if x_method == "gleams" else ""
    if irrelevant:
        explanation_filename = "_".join([dataset, model, "irrelevant", f"K{relevant_vars}", x_method, params_in_name]) + ".pkl"
    else:
        if local:
            explanation_filename = "_".join([dataset, model, x_method, params_in_name,"local"]) + ".pkl"
        else:
            explanation_filename = "_".join([dataset, model, x_method, params_in_name, "global"]) + ".pkl"
    explanation_path_ = os.path.join(explanation_folder, explanation_filename)
    return explanation_path_


def model_path(dataset, model, irrelevant=False,relevant_vars=None, is_keras=False):
    """Get the Model Path where to store computed model or to load an already existing model"""
    explanation_folder = "./explanations"

    if irrelevant and relevant_vars is None:
        raise Exception("When saving IrrelevantVarsRegressor, you must pass the number of relevant vars")
    if irrelevant:
        model_filename = "_".join([dataset, model, "relevant_vars", f"K{relevant_vars}"])
    model_filename = model_filename + ".h5py" if is_keras else model_filename + ".pkl"
    model_path_ = os.path.join(explanation_folder, model_filename)
    return model_path_

def load_previous_model(model_path, is_keras=False, zip=False):
    """Load pickled models"""

    path_to_be_tested = model_path+".zip" if zip and is_keras else model_path
    if os.path.exists(path_to_be_tested):
        if is_keras:
            return load_nn(model_path, zip=zip)
        else:
            return load_pickle_object(model_path)
    else:
        return None


def load_previous_expl(x_methods, custom_explanation_path_function):
    attributions = {x_method: None for x_method in x_methods}

    for x_method in x_methods:
        if os.path.exists(custom_explanation_path_function(x_method)):
            attributions[x_method] = load_pickle_object(custom_explanation_path_function(x_method))
        else:
            pass
    return attributions


def run_monotonicity_test(dataset, model, n_mc, n_sobol_points):
    """

    Computing the monotonicity metric from Nguyen and Martinez paper (https://arxiv.org/pdf/2007.07584.pdf):
     Spearman's correlation between the absolute values of attributions and expected loss
    when removing features (Eq. (1) of the paper).

    :return: two dictionaries containing the local and global monotonicity for the chosen explanation methods
    """

    np.random.seed(42)
    print(f"Compute Monotonicity on {dataset=}")
    x_methods = ["lime", "shap", "gleams", "pdp"]

    # parameters check
    filename = _check_params(dataset, model)
    is_keras = True if model == "nn" else False
    file_path = os.path.join(os.path.join("./data", dataset), filename)
    model_name = "_".join([dataset, model, "model"])
    ml_model_path = os.path.join(r"./models", model_name)

    def mon_explanation_path_global(x_method):
        return explanation_path(x_method, dataset, model, n_sobol_points, irrelevant=False, local=False)

    def mon_explanation_path_local(x_method):
        return explanation_path(x_method, dataset, model, n_sobol_points, irrelevant=False, local=True)

    # import dataset and split into train/test
    X, y = import_preproc_dataset(dataset=dataset, file_path=file_path)
    n_points, n_dims = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"{X_test.shape=}")
    # # TODO: remove the X_test, y_test restriction
    # X_test = X_test[:200]
    # y_test = y_test[:200]

    # load ML model
    if model == "xgb":
        regressor = load_previous_model(ml_model_path + ".pkl",is_keras=False)
    elif model == "nn":
        regressor = load_previous_model(ml_model_path + ".h5py",is_keras=True,zip=True)

    # TODO: if not already stored, train model xgb or nn

    ## Global Attributions
    global_attributions = load_previous_expl(x_methods=x_methods, custom_explanation_path_function=mon_explanation_path_global)

    # Gleams global
    if os.path.exists(f"./explanations/Gleams_Model_{dataset}_{model}_sob{n_sobol_points}.pkl"):
        gleams_glob_exp = load_pickle_object(f"./explanations/Gleams_Model_{dataset}_{model}_sob{n_sobol_points}.pkl")
    else:
        gleams_glob_exp = train_gleams(n_sobol_points=n_sobol_points, model=regressor,
                                     X_data=X_train, is_keras=is_keras)
        save_pkl_model(my_object=gleams_glob_exp, is_gleams_nn=is_keras,
                       path=f"./explanations/Gleams_Model_{dataset}_{model}_sob{n_sobol_points}.pkl")

    if global_attributions["gleams"] is None:
        print("GLobal Monotonicity GLEAMS")
        # gleams_glob_exp = train_gleams(n_sobol_points=n_sobol_points, model=regressor,
        #                              X_data=X_train, is_keras=is_keras)
        # fig = gleams_glob_exp.global_importance(true_to="model", meaning="average_impact", show=False, save=True)
        # fig = gleams_glob_exp.global_importance(true_to="data", data=X_train, meaning="average_impact", show=False, save=True, path="./global_importance_true_to_data.pdf")
        # fig = gleams_glob_exp.local_importance(X_test.iloc[42], show=False, standardized=True, save=True)
        # time.sleep(60)
        # fig = gleams_glob_exp.local_importance(X_test.iloc[42], show=False, standardized=False, save=True)

        global_attributions["gleams"] = np.zeros(X_test.shape)
        for i in tqdm(range(X_test.shape[0]), desc="Computing GLEAMS Global What-If Importance on Test Points"):
            example = X_test.values[i]
            global_attributions["gleams"][i] = gleams_glob_exp.whatif_global_importance(example, standardize="global")
        # save_pickle_object(global_attributions["gleams"], mon_explanation_path_global("gleams"))

    # PDP global
    if global_attributions["pdp"] is None:
        print("GLobal Monotonicity PDP")
        global_attributions["pdp"] = train_pdp(regressor, X_test, model_type=model)
        save_pickle_object(global_attributions["pdp"], mon_explanation_path_global("pdp"))

    # Shap global
    if global_attributions["shap"] is None:
        print("GLobal Monotonicity SHAP")
        global_attributions["shap"] = train_shap(model=regressor, X_data=X_test, model_type=model)
        save_pickle_object(global_attributions["shap"], mon_explanation_path_global("shap"))

    # Lime global
    if global_attributions["lime"] is None:
        print("GLobal Monotonicity LIME")
        global_attributions["lime"] = np.zeros(X_test.shape)
        lime_explainer = get_lime(train_data=X_test, cat_indices=get_lime_cat_indices(dataset))
        for i in tqdm(range(X_test.shape[0]), desc="Computing LIME on Test Points"):
            example = X_test.values[i]
            global_attributions["lime"][i] = train_lime(lime_explainer, example, regressor.predict, num_features=n_dims)
        save_pickle_object(global_attributions["lime"], mon_explanation_path_global("lime"))

    # Local (most populous leaf) and Global boundaries
    max_leaf = get_most_crowded_leaf(gleams_glob_exp.mob, X_test.values)
    local_boundaries = dict(max_leaf.domain)
    global_boundaries = gleams_glob_exp.mob.domain_dict
    # get points in the max_leaf
    local_ids = check_domain(X_test.values, max_leaf)
    n_local_test_points = len(local_ids)
    local_points = X_test.iloc[local_ids]
    local_points.reset_index(drop=True, inplace=True)
    print(f"Local Monotonicity done on {n_local_test_points} of the test points")
    fig = gleams_glob_exp.local_importance(local_points.iloc[0], standardized=True, show=False, save=True)

    ## Local Attributions
    local_attributions = load_previous_expl(x_methods=x_methods, custom_explanation_path_function=mon_explanation_path_local)

    # Gleams local
    if local_attributions["gleams"] is None:
        print("Local Monotonicity GLEAMS")
        attribution_dict = gleams_glob_exp.local_importance(local_points.iloc[0], standardized=True, show=False)[0]["coefficients"]
        local_attributions["gleams"] = [attribution_dict[var] for var in local_points.iloc[0].index]

        # save_pickle_object(local_attributions["gleams"], mon_explanation_path_local("gleams"))

    # PPD local
    if local_attributions["pdp"] is None:
        print("Local Monotonicity PDP")
        local_attributions["pdp"] = train_pdp(regressor, local_points, model_type=model)
        save_pickle_object(local_attributions["pdp"], mon_explanation_path_local("pdp"))


    # Compute global and local monotonicity
    global_monotonicity = {x_method: list() for x_method in x_methods}
    local_monotonicity = {x_method: list() for x_method in x_methods}

    for i in tqdm(range(X_test.shape[0]), desc="Computing Monotonicity over the Test Points"):

        example = X_test.values[i]
        global_e = compute_single_example_e_list(example=example, model=regressor, mc_boundaries=global_boundaries,
                                                 n_mc=n_mc,is_keras=is_keras)
        single_global_attributions = {"lime": global_attributions["lime"][i],
                                      "shap": global_attributions["shap"][i],
                                      "pdp": global_attributions["pdp"],
                                      "gleams": global_attributions["gleams"][i]}

        for x_method in x_methods:
            # print(f"\n{x_method=}")
            global_monotonicity[x_method].append(compute_single_monoton(global_e, single_global_attributions[x_method]))

        # local monotonicity
        if i in local_ids:
            local_e = compute_single_example_e_list(example=example, model=regressor, mc_boundaries=local_boundaries,
                                                    n_mc=n_mc,is_keras=is_keras)
            single_local_attributions = {"lime": global_attributions["lime"][i],
                                          "shap": global_attributions["shap"][i],
                                          "pdp": local_attributions["pdp"],
                                          "gleams": local_attributions["gleams"]}

            for x_method in x_methods:
                # print(f"\n{method=}")
                local_monotonicity[x_method].append(compute_single_monoton(local_e, single_local_attributions[x_method]))

    # aggregate monotonicity scores of single test units into final monotonicity value
    global_monotonicity = {x_method: np.mean(global_monotonicity[x_method]) for x_method in x_methods}
    local_monotonicity = {x_method: np.mean(local_monotonicity[x_method]) for x_method in x_methods}

    return local_monotonicity, global_monotonicity


def run_recall_test(dataset, model, n_relevant_vars, n_sobol_points):
    """


     As per feature attributions of the explanation method,
     we create a vector with 0 value for irrelevant vars, 1 for the other features
    (this should be a reasonable attribution vector)

    :return: the monotonicity value for the first test instance
    """

    np.random.seed(42)
    x_methods = ["lime", "shap", "gleams", "pdp"]

    # parameters check
    filename = _check_params(dataset, model)

    print(f"Using {dataset=}")
    data_path = os.path.join(os.path.join("./data", dataset), filename)
    model_name = "_".join([dataset, model, "model"])
    explanation_folder = "./explanations"

    def rec_explanation_path(x_method):
        return explanation_path(x_method, dataset, model, n_sobol_points, irrelevant=True, relevant_vars=n_relevant_vars)

    # import dataset and split into train/test
    X, y = import_preproc_dataset(dataset=dataset, file_path=data_path)
    # X,y = X.values, y.values
    n_points, n_dims = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    n_irrelevant_vars = n_dims - n_relevant_vars

    # #TODO: remove the X_test, y_test restriction
    # X_test = X_test[:200]
    # y_test = y_test[:200]

    if model == "xgb":
        regressor = get_xgb_regr(y_train.mean(), learning_rate=0.01)
        fit_params = {"early_stopping_rounds": 300,
                      "eval_set": [(X_test, y_test)],
                      "eval_metric": "rmse",
                      "verbose": 1000}
        is_keras = False

    elif model == "nn":

        min_delta_params = {"parkinson": 0.3,
                     "houses": 3,
                     "wine": 0.03}

        regressor = get_nn_regr(n_vars=X_train.shape[1] - n_irrelevant_vars, learning_rate=0.005)
        fit_params = {"epochs": 40000,
                      "validation_data": [(X_test, y_test)],
                      "callbacks": [EarlyStopping_verbose(patience=700,
                                                          min_delta=min_delta_params[dataset],
                                                          nepochs=500)],
                      "verbose": 0,
                      "batch_size": len(X_train),
                      }
        is_keras = True


    # instanziate and fit the IrrelevantVarsRegressor
    print(f"Train Irrelevant Regressor, K{n_relevant_vars}")
    irrelevant_vars_model = IrrelevantVarsRegressor(model=regressor)
    irrelevant_vars_model.fit(X_train, y_train, n_irrelevant_ids=n_irrelevant_vars, **fit_params)

    # load stored attributions
    global_attributions = load_previous_expl(x_methods=x_methods, custom_explanation_path_function=rec_explanation_path)

    # compute and store attributions if there were no stored files
    if global_attributions["gleams"] is None:
        print("Recall GLEAMS")
        gleams_glob_exp = train_gleams(n_sobol_points=n_sobol_points, model=irrelevant_vars_model,
                                     X_data=X_train, is_keras=is_keras)
        global_attributions["gleams"] = np.zeros(X_test.shape)
        for i in tqdm(range(X_test.shape[0]), desc="Computing GLEAMS Global What-If Importance on Test Points"):
            example = X_test.iloc[i]
            attribution_dict = gleams_glob_exp.local_importance(sample=example, standardized=True, show=False)[0]["coefficients"]
            global_attributions["gleams"][i] = [attribution_dict[var] for var in example.index]
        # save_model(my_object=gleams_glob_exp, is_gleams_nn=is_keras,
        #            path=rec_explanation_path("gleams")[:-4] + "_expl_model.pkl")
        # save_pickle_object(global_attributions["gleams"], rec_explanation_path("gleams"))

    if global_attributions["pdp"] is None:
        print("Recall PDP")
        global_attributions["pdp"] = train_pdp(irrelevant_vars_model, X_test, model_type=model)
        save_pickle_object(global_attributions["pdp"], rec_explanation_path("pdp"))

    if global_attributions["shap"] is None:
        print("Recall SHAP")
        global_attributions["shap"] = train_shap(model=irrelevant_vars_model, X_data=X_test, model_type=model,
                                       irrelevant_regr=True)
        if isinstance(global_attributions["shap"], list):
            global_attributions["shap"] = global_attributions["shap"][0]
        save_pickle_object(global_attributions["shap"], rec_explanation_path("shap"))

    if global_attributions["lime"] is None:
        global_attributions["lime"] = np.zeros(X_test.shape)
        print("Recall LIME")
        lime_explainer = get_lime(train_data=X_test,cat_indices=get_lime_cat_indices(dataset))
        for i in tqdm(range(X_test.shape[0]), desc="Processing Test Points"):
            example = X_test.values[i]
            global_attributions["lime"][i] = train_lime(lime_explainer, example, irrelevant_vars_model.predict,
                                              num_features=n_dims)
        save_pickle_object(global_attributions["lime"], rec_explanation_path("lime"))

    # Compute true_attributions vector (1 for relevant features, 0 for irrelevant)
    relevant_vars_ids = list(set(range(n_dims)) - set(irrelevant_vars_model.irrelevant_vars_ids))
    true_attributions = [1 if id_var in relevant_vars_ids else 0 for id_var in range(n_dims)]



    # compute recall for all the explanation methods (one test unit at a time, then average the predictions)
    recall_single_examples = {x_method: list() for x_method in x_methods}

    for i in tqdm(range(X_test.shape[0]), desc="Computing Recall over the Test Points"):
        if i in range(global_attributions["shap"].shape[0]):
            single_global_attributions = {"lime": global_attributions["lime"][i],
                                          "shap": global_attributions["shap"][i],
                                          "pdp": global_attributions["pdp"],
                                          "gleams": global_attributions["gleams"][i]}
        else:
            single_global_attributions = {"lime": global_attributions["lime"][i],
                                          "shap": None,
                                          "pdp": global_attributions["pdp"],
                                          "gleams": global_attributions["gleams"][i]}
        for x_method in x_methods:
            if single_global_attributions[x_method] is None:
                pass
            else:
                recall_single_examples[x_method].append(compute_recall(true_attributions, single_global_attributions[x_method],
                                                  n_irrelevant_vars=n_irrelevant_vars))

    recall = {x_method: "Not Computed yet" for x_method in x_methods}
    recall_std = {x_method: "Not Computed yet" for x_method in x_methods}
    for x_method in x_methods:
        recall[x_method] = np.mean(recall_single_examples[x_method])
        recall_std[x_method] = np.std(recall_single_examples[x_method])

    return recall, recall_std


def run_tests(dataset,n_sobol_points,relevant_vars_list, n_mc=1000):

    output_file_path = os.path.join("./results",f"results_{dataset}_sob{n_sobol_points}.xlsx")
    results = dict()


    # for model in ["nn"]:
    for model in ["xgb","nn"]:

        print(f"Results for {model}")
        local_mon,global_mon = run_monotonicity_test(dataset=dataset, model=model,n_mc=n_mc,
                                                     n_sobol_points=n_sobol_points)
        print(f"{local_mon=},{global_mon=}")

        recall = np.zeros((len(relevant_vars_list),4))
        recall_std = np.zeros((len(relevant_vars_list),4))
        for id_row,n_relevant_vars in enumerate(relevant_vars_list):
            rec, rec_std = run_recall_test(dataset=dataset, model=model, n_sobol_points=n_sobol_points,
                                  n_relevant_vars=n_relevant_vars)
            print(f"{rec=}")
            print(f"{rec_std=}")

            rec_df, rec_std_df = pd.Series(rec, index=rec.keys()),pd.Series(rec_std, index=rec_std.keys())
            colnames_recall = rec_df.index
            recall[id_row] = rec_df
            recall_std[id_row] = rec_std_df

        local_mon = pd.Series(local_mon,index=local_mon.keys())
        global_mon = pd.Series(global_mon,index=global_mon.keys())
        recall_df = pd.DataFrame(recall,columns=colnames_recall,index=[f"recall_K{n}" for n in relevant_vars_list])
        recall_std_df = pd.DataFrame(recall_std, columns=colnames_recall, index=[f"recall_stdK{n}" for n in relevant_vars_list])
        monoton_df = pd.DataFrame([local_mon,global_mon],index=["local_monotononicity","global_monotononicity"])
        results[f"{model}_sobol{n_sobol_points}"] = pd.concat([monoton_df,recall_df, recall_std_df],axis=0)
        # results[f"{model}_sobol{n_sobol_points}"] = pd.concat([recall_df, recall_std_df],axis=0)

    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, res in results.items():
            res.to_excel(writer, sheet_name=sheet_name)



if __name__ == "__main__":

    # run_tests(dataset="wine",n_sobol_points=15,relevant_vars_list=[6,10])
    run_tests(dataset="houses",n_sobol_points=17, relevant_vars_list=[10])
    # run_tests(dataset="parkinson",n_sobol_points=15,relevant_vars_list=[10])
