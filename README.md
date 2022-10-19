# GLEAMS: Bridging the Gap Between Local and Global Explanations


## TODO: say experiments are roughly the same because of initialization differences...

TODO: to write stuff here

## Steps to reproduce the code on Windows OS:

- Open the terminal from the Gleams folder
- Activate the gleams_venv by using ".\gleams_venv\Scripts\activate"
- navigate to Experiments/
- Run the evaluation.py file using the command python evaluation.py

If you do not have Windows, the steps to reproduce the code are the same, but the actual lines of code might be different. Please refer to [venv documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to check out how to properly run the code on your OS.


## Black-Box Models Specifics:

- Gradient Boosting model: employed the XGBoost framework, with booster='gbtree' (corresponding to the basic CART Tree), random_state=42, max_depth=2, learning_rate=0.05, n_estimators=40000 (never reached since we use early stopping, setting early_stopping_rounds=100, eval_metric='rmse')

- Neural Network: employed the keras package to build a Multi Layer Perceptron (MLP) of 3 layers (2 Dense layers of 264 neurons each with sigmoid activation, 1 Dense layer of a single neuron with linear activation to obtain the predictions). We optimize the 'MSE' metric, using the Adam optimizer with initial learning_rate=0.005.
The number of epochs is 40000, never reached since we use early stopping, setting patience=700. Regarding the min_delta parameter we use a different value for each dataset, since the target variable has different scales of measurement. In particular, min_delta has value 0.03 for the wine dataset, 0.3 for parkinson, 3 for houses.