# GLEAMS: Bridging the Gap Between Local and Global Explanations
#### Supplementary material

## About Paper Experiments

The experiments to reproduce the results in Table 2 of the Paper, are contained in Experiments/evaluation.py.  

***Warning I***: results may slightly vary due to randomness in the initialization and training of both the Machine Learning models and explanation models (eg. LIME generates 5000 random datapoints, Shap randomly samples a number of variable combinations, etc).  

***Warning II***: experiments require long computation time to run. Just to give an idea, on a Intel-i9 11th gen CPU, 32GB RAM, 8 cores Laptop the evaluation.py script takes approximately 52 hours to run. 


## Black-Box Models Specifics:

- Gradient Boosting model: employed the XGBoost framework, with booster='gbtree' (corresponding to the basic CART Tree), random\_state=42, max\_depth=2, learning\_rate=0.05, n\_estimators=40000 (never reached since we use early stopping, setting early\_stopping\_rounds=100, eval\_metric='rmse')

- Neural Network: employed the keras package to build a Multi Layer Perceptron (MLP) of 3 layers (2 Dense layers of 264 neurons each with sigmoid activation, 1 Dense layer of a single neuron with linear activation to obtain the predictions). We optimize the 'MSE' metric, using the Adam optimizer with initial learning\_rate=0.005.
The number of epochs is 40000, never reached since we use early stopping, setting patience=700. Regarding the min\_delta parameter we use a different value for each dataset, since the target variable has different scales of measurement. In particular, min\_delta has value 0.03 for the wine dataset, 0.3 for parkinson, 3 for houses.

## Steps to reproduce the code (specific for Windows OS):

- Open the terminal from the AISTATS\_submission\_code folder
- download virtualenv package ```py -m pip install --user virtualenv```
- Create your own virtual environment ```py -m venv env```
- Activate the gleams_venv by using ```.\env\Scripts\activate```
- Install required packages through requirements.txt ```py -m pip install -r requirements.txt```
- navigate to the Experiments folder, using ```cd Experiments```
- Run the evaluation.py main file to obtain the results of the paper, using ```python -O evaluation.py``` (the ```-O``` option is required to avoid Shap raising errors due to unit tests failure)  


If you do not have a Windows machine, the steps to reproduce the code are the same, but the actual lines of code might be different. Please refer to [venv documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to check out how to properly run the code on your OS.


