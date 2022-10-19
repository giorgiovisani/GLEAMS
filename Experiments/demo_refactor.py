import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from evaluation import load_previous_model
from sklearn.model_selection import train_test_split
from ml_models import _check_params, import_preproc_dataset, load_nn, load_pickled_object, save_pkl_model, save_keras_zip_model, get_xgb_regr, get_nn_regr, EarlyStopping_verbose
from explanations import train_gleams

dataset = "wine"
model = "xgb"
n_sobol_points = 12

filename = _check_params(dataset, model)
is_keras = True if model == "nn" else False
file_path = os.path.join(os.path.join("./data", dataset), filename)
model_name = "_".join([dataset, model, "model"])
ml_model_path = os.path.join(r"./models", model_name)

X, y = import_preproc_dataset(dataset=dataset, file_path=file_path)
n_points, n_dims = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"{X_test.shape=}")

# load ML model
if model == "xgb":
    regressor = load_previous_model(ml_model_path + ".pkl", is_keras=False)
elif model == "nn":
    regressor = load_previous_model(ml_model_path + ".h5py", is_keras=True, zip=True)

gleams_glob_exp = train_gleams(n_sobol_points=n_sobol_points, model=regressor,
                               X_data=X_train, is_keras=is_keras)

single_sample = X_test.values[42]
fig = gleams_glob_exp.global_importance(true_to="model", meaning="average impact", show=False, save=False)
fig = gleams_glob_exp.global_importance(true_to="model", meaning="ranking importance", show=False, save=False)
fig = gleams_glob_exp.global_importance(true_to="data", data=X_train, meaning="average impact", show=False, save=False)
fig = gleams_glob_exp.global_importance(true_to="data", data=X_train, meaning="ranking importance", show=False,
                                        save=False)
fig = gleams_glob_exp.local_importance(single_sample, show=False, standardized=True, save=False)
fig = gleams_glob_exp.local_importance(single_sample, show=False, standardized=False, save=False)
attributions = gleams_glob_exp.whatif_global_importance(single_sample, standardize="global")
attributions = gleams_glob_exp.whatif_global_importance(single_sample, standardize="local")
attributions = gleams_glob_exp.whatif_global_importance(single_sample, standardize=None)

save_pkl_model(gleams_glob_exp, is_gleams_nn=is_keras, path="./prova.pkl", force=True, compressed=False)
gleams_glob_exp_loaded = load_pickled_object("./prova.pkl")
gleams_glob_exp_loaded.check_performance()
