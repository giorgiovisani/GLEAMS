#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import zipfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam
from keras.saving.save import save_model
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from utils import load_pickle_object, save_pickle_object


class EarlyStoppingVerbose(Callback):
    """
    My own EarlyStopping class with the right verbosity
    inspired by:
    https://stackoverflow.com/questions/69635182/capturing-epoch-count-when-using-earlystopping-feature-with-keras-model

    """

    def __init__(self, patience=0, nepochs=1, min_delta=0.01):
        super(EarlyStoppingVerbose, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.nepochs = nepochs
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        if (int(epoch) % self.nepochs) == 0:
            print("Epoch: {:>3}".format(
                epoch) + " | Valid loss rmse (best epoch till now): " + f"{np.sqrt(self.best):.4e}")

        current = logs.get("val_loss")
        if np.less(np.sqrt(current), np.sqrt(self.best) - self.min_delta):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Final EarlyStopping Epoch: {:>3}".format(
                self.stopped_epoch) + " | Best epoch loss-rmse: " + f"{np.sqrt(self.best):.4e}")


def _check_params(dataset, model=None):
    """
    Check input parameters for train_ml_model()

    :param dataset:
    :param model:
    :return:
    """

    if dataset not in ["houses", "wine", "parkinson"]:
        raise Exception("Non valid dataset")
    if model:
        if model not in ["xgb", "nn"]:
            raise Exception("Non valid model")

    file_dict = {
        "wine": "winequality-white.csv",
        "houses": "kc_house_data.csv",
        "parkinson": "parkinsons_updrs.data",
    }
    return file_dict[dataset]


def import_preproc_wine(file_path):
    """Import and Preprocess - Wine Dataset"""
    df = pd.read_csv(file_path, sep=";")
    y = df.pop("quality")
    return df, y


def import_preproc_parkinson(file_path):
    """Import and Preprocess - Parkinson Dataset
    Inspired by: https://www.kaggle.com/code/mountainguest/parkinson-telemonitoring-regression-with-keras/data?select=parkinsons_updrs.data"""
    df = pd.read_csv(file_path, sep=",")
    y = df.pop("total_UPDRS")
    df.pop("motor_UPDRS")
    df.pop("subject#")
    return df, y


def import_preproc_houses(file_path):
    """Import and Preprocess - Houses Dataset

    Inspired by: https://www.kaggle.com/code/bansodesandeep/keras-regression-code-along-project
    Price variable is rescaled by 10K
    """
    df = pd.read_csv(file_path, sep=",")

    df = df.drop('id', axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].apply(lambda date: date.month)
    df['year'] = df['date'].apply(lambda date: date.year)
    df = df.drop('date', axis=1)
    df = df.drop('zipcode', axis=1)
    y = df.pop("price") / 10000
    return df, y


def import_preproc_dataset(dataset, file_path):
    if dataset == "wine":
        X, y = import_preproc_wine(file_path)
    elif dataset == "houses":
        X, y = import_preproc_houses(file_path)
    elif dataset == "parkinson":
        X, y = import_preproc_parkinson(file_path)

    return X, y


def save_keras_zip_model(model, path="./nn_model"):
    """
    Save the Keras/TF NN model as hdf5.zip file.

    Parameters
    ----------
    path: Name of the file to be saved.
    Returns
    -------
    None
    """

    # save the nn model (as a hdf5 zip)

    path += ".h5py"
    save_model(model)
    if os.path.isfile(path + ".zip"):
        os.remove(path + ".zip")
    with ZipFile(path + ".zip", 'w') as myzip:
        myzip.write(path)
    # model.save(path, save_format="h5")
    # shutil.make_archive(path, "zip", "./")
    # shutil.rmtree(path)
    os.remove(path)


def get_xgb_regr(y_mean, learning_rate=0.01):
    regressor = XGBRegressor(
        booster="gbtree",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        base_score=y_mean,
        n_estimators=40000,
        learning_rate=learning_rate,
        max_depth=2,
        use_label_encoder=False
    )
    return regressor


def fit_xgb_regr(xgb_regr, X_train, X_val, y_train, y_val):
    xgb_regr.fit(
        X_train, y_train,
        early_stopping_rounds=100,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        verbose=1000)


def get_nn_regr(n_vars, learning_rate=0.005):
    regressor = Sequential()
    regressor.add(Dense(264, input_shape=(n_vars,), activation='sigmoid', name='dense_1'))
    regressor.add(Dense(264, activation='sigmoid', name='dense_2'))
    regressor.add(Dense(1, activation='linear', name='dense_output'))
    regressor.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse'])
    # print(regressor.summary())
    return regressor


def fit_nn_regr(nn_regr, X_train, X_val, y_train, y_val, patience=700, min_delta=0.01):
    history = nn_regr.fit(X_train, y_train, epochs=40000,
                          validation_data=(X_val, y_val),
                          verbose=0,
                          callbacks=[EarlyStoppingVerbose(patience=patience, min_delta=min_delta, nepochs=500)],
                          batch_size=len(X_train)
                          )
    return history


def save_pkl_model(my_object: object, is_gleams_nn: bool = False, path: str = "./model", force: bool = False,
                   compressed: bool = False) -> None:
    if is_gleams_nn:
        my_object.predict_function = None

    save_pickle_object(my_object, path, force, compressed)
    return None


def load_pickled_object(model_path):
    """Load a pickled model"""
    regressor = pickle.load(open(model_path, 'rb'))
    return regressor


def load_nn(model_path, zip=True):
    """Unzip the NN file, load it and remove the unzipped version from the os
    """
    if zip:
        with zipfile.ZipFile(model_path + ".zip", 'r') as zip_ref:
            zip_ref.extractall("./")
    regressor = load_model(model_path)
    if zip:
        os.remove(model_path)
    return regressor


def load_previous_model(model_path, is_keras=False, zip=False):
    """Load stored black-box models"""

    path_to_be_tested = model_path + ".zip" if zip and is_keras else model_path
    if os.path.exists(path_to_be_tested):
        if is_keras:
            return load_nn(model_path, zip=zip)
        else:
            return load_pickle_object(model_path)
    else:
        return None


def train_ml_model(dataset, model, nn_learning_rate, patience, min_delta):
    """
    Train the given ML model (specified in the 'model' parameter) on the requested dataset,
    with specified parameters
    """

    # parameters check
    filename = _check_params(dataset, model)

    print(f"Using {dataset=}")
    data_folder = "./data"
    model_name = "_".join([dataset, model, "model"])
    model_path = os.path.join(r"./models", model_name)

    file_path = os.path.join(os.path.join(data_folder, dataset), filename)

    X, y = import_preproc_dataset(dataset=dataset, file_path=file_path)

    # # store cleaned data as csv (to be used in the dashboard)
    # data = pd.concat([X, y], axis=1)
    # data.to_csv(f"./data/{dataset}/{filename[:-4]}_right.csv",index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    if model == "xgb":

        regressor = get_xgb_regr(y_train.mean(), learning_rate=0.05)
        fit_xgb_regr(regressor, X_train, X_test, y_train, y_test)
        print("Proper number of trees: {}".format(regressor.best_iteration))
        save_pickle_object(regressor, path=model_path + ".pkl")

    elif model == "nn":

        regressor = get_nn_regr(n_vars=X_train.shape[1], learning_rate=nn_learning_rate)
        history = fit_nn_regr(regressor, X_train, X_test, y_train, y_test, patience=patience, min_delta=min_delta)
        save_keras_zip_model(regressor, path=model_path)


def train_all():
    """Train XGBoost and NN on all the three datasets and store the models"""
    for model in ["xgb", "nn"]:
        train_ml_model(dataset="wine", model=model, nn_learning_rate=0.005, patience=700, min_delta=0.03)
        train_ml_model(dataset="houses", model=model, nn_learning_rate=0.005, patience=700, min_delta=3)
        train_ml_model(dataset="parkinson", model=model, nn_learning_rate=0.005, patience=700, min_delta=0.3)


if __name__ == "__main__":
    train_all()
