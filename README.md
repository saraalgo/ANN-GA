# Artificial Neural Network trained with Genetic Algorithm Model

This repository is already functional with the current initial version (v0.0).

## About

The main objective of this repository is to provide an unified Machine Learning (ML) model design whose both employs Artificial Neural Networks (ANN) and Genetic Algorithm (GA). By using the following descripted pipeline, it is possible to create a model with an ANN which uses [Tensorflow](https://github.com/tensorflow/tensorflow) methods to its building and it is trained to be optimized by the GA implemented in the package [Geneticalgorithm2](https://github.com/PasaOpasen/geneticalgorithm2).


## Prerequisites

1. Install python 3.7.9

## Installation

1. Clone the repository in your personal device using the following command:

```sh
git clone https://github.com/saraalgo/ANN-GA.git
```

2. Create and activate python environment if you do not count with the beforehand mentioned Python version. Otherwise, you could skip this step.

```sh
python3.7.9 -m venv ANN-GA/
source bin/activate
```

3. Upgrade pip and install project requirements 
```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project workflow

1. In *params.py*:
  - First define **params**
  - Then **ga_params**
  - Finally **ann_params**
2. Run the the final function defined in *ga_ann.py*:
```python
model = GA_ANN()
```

### Input Data format

The data you would decide to use as Dataset to train and predict for your model **MUST**:
- Be on *.csv* format. 
- The **first row** of the file contains the **header** of the columns.
- Each row starting from the second will be an observation.
- All the data is **numeric**.


### Params configuration

First of all, it is essential to modify the [params.py](https://github.com/saraalgo/ANN-GA/blob/main/src/params.py) which contains the three dictionary with all the information it is possible to personalize without modify the code.

#### Params

The first dictionary is called **params**, the default values are the following:

```python
params = {
    "inputfile": "../data/data.csv",
    "input_name": "Data",
    "normalize_data": True,
    "test_size": 0.3,
    "output_colname": "output",
    
    "fitness_metric": "loss",
    "var_type": "real",
    "timeout": 5000
}
```
* *inputfile*: Here is where you must indicate where your dataset *.csv* file is located, starting your path from the folder **/src**.
* *input_name*: Name to save the results of your model.
* *normalize_data*: In case your data is not normalized, the default option is to normalize it between [-1 1] due to be recommended for ANN models.
* *test_size*: Proportion of the data to use as test after having trained with the *1-test_size* proportion part.
* *output_colname*: Name of the columns which provides the output of the dataset.
* *fitness_metric*: Metric used as **fitness** for the GA. Available: *loss*, *accuracy*, *auc*, *precision*, *recall* and *prc*.
* *var_type*: GA params before run it, indicates the kind of data will compose de genome of the individuals of the new population.
* *timeout*: Another GA param before run it, time to wait until GA start running function. If the time exceeds this variable, it will raise an error.

#### GA params

Next, the **ga_params** dictionary are displayed, with the following default values:

```python
ga_params = {
    "max_num_iteration": 4,
    "population_size": 10,
    "mutation_probability": 0.1, 
    "elit_ratio": 0.01,
    "parents_portion": 0.3,
    "crossover_type": "uniform",
    "max_iteration_without_improv": 50  
}
```
* *max_num_iteration*: Number of individuals on each generation.
* *population_size*: Maximun number of generations.
* *mutation_probability*: Proportion of mutation for the individuals creating the next generation.
* *elit_radio*: Proportion of elitism used creating the next generation.
* *parents_portion*: Proportion of the individuals for the next generations belonging to the parents pool.
* *crossover_type*: Type of crossover to cross parents. Available: "one_point", "two_point", "uniform", "segment", "shuffle"... More info in [Geneticalgorithm2](https://github.com/PasaOpasen/geneticalgorithm2/blob/master/geneticalgorithm2/crossovers.py) package.
* *max_iteration_without_improv*: Stopping criteria value that indicates the number of generations to stop the evolution if the fitness value does not improve.

#### ANN params

Finally, the **ann_params** must be defined, the below values are the default ones:

```python
ann_params = {
    "Input layer": 18,
    "Hiddens layers": [12,6],
    "Output layer": 1,

    "Activation Function": "relu",

    "Loss": "binary_crossentropy",
    
    "Optimizer": "sgd", 
    "Optimizer params" : "learning_rate",
    "Optimizer params learning_rate": 0.01,

    "Verbose": 0,
    "Batch size": 125
}
```

* *Input layer*: Number of neurons for the input layer.
* *Hiddens layers*: Number of neurons in the hidden layers. It is possible to stablish not only the number of neurons, but also increment the number of hidden layers as much as you need. Minimum of one hidden layer.
* *Output layer*: Number of neurons in the last layer
* *Activation Function*: Activation function to use between layers in the ANN. Available: "relu", "ADAM"... More info in [Tensorflow activation functions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/activations.py).
* *Loss*: Loss function selected for ANN compilation. Recommended: "binary_crossentropy" for *binary* problems, "mse" for *regression* problems and "categorical_crossentropy" for *Multiclass* problems.
* *Optimizer*: Optimizer selected for compile the ANN. More info about available options in [Tensorflow optimizers](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py).
* *Optimizer params*: To modify params inside the previously selected optimizer, you can add the name of the param to modify here. 
* *Optimizer params XXX*: For each param chosen to modify, it is compulsary to add a line with de value selected.

For *"Optimizer params": XXX* and *"Optimizer params XXX": Value* it is possible to modify more than one at once introducing them inside [] and the values in diferent lines. For instance with [Adam optimizer](https://keras.io/api/optimizers/adam/): 

```python
"Optimizer": "Adan", 
"Optimizer params" : ["learning_rate", "epsilon", "beta_1"],
"Optimizer params learning_rate": 0.01,
"Optimizer params epsilon": 1e-07,
"Optimizer params beta_1": 0.9,
```

* *Verbose*: In the evaluation of the ANN, option of how much to display that function. Available from 0 to 2, being 0 no display at all, to the maximun for that Tensorflow option.

* *Batch size*: Number of samples per batch of computation. If unspecified, batch_size will default to 32. Also a Tensorflow param essential for ANN evaluation.


### How to run

1. Be aware to stablish the current path in the **/src** folder. In case Python is opened in the previous explained environment, follow the next commands:

```python
import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH + "/src/")
```

2. **Load** both the previously modified params and the script with the ML model function, *params.py* and *ga_ann.py* respectively.

```python
from ga_ann import *
from params import *
```

3. **Run** the model funtion called *GA_ANN*. This function will build your ANN, create your individuals and carry out the designed GA to train it with the loaded data. This command will also create a folder called **/data/XXX**, being "XXX" the value which indicates the name of the experiment (inside *params* dictionary, "input_name" option). A **/tmp** folder will be also automatically created to be able to extract data calculated inside the GA which can be printed and saved with the below explained functions.

```python
model = GA_ANN()
```

4. Functions that can display or save params and results obtained after the GA:

* *get_data()*: Return the data in all the steps it followed

```python
data, X, y, X_train, y_train, X_test, y_test = model.get_data()
```
  
  * data: The loaded *.csv*.
  * X: The features for the model.
  * y: The output column.
  * X_train: X proportion to train.
  * X_test: X proportion to test.
  * y_train: output values associated to the X_train.
  * y_test: output values associated to the y_test.

* *get_result*: Return the dictionary created by default with the Geneticalgorithm2 package.

```python
result = model.get_result()
```

* *get_bests_metrics*: This function will return a pandas DataFrame with the metrics of train and test for the best individual on each generation. 

```python
bests = model.get_bests_metrics()
```
This DataFrame will have a total of 26 columns:
- "generation": number of generation (from 1 to max achieved)
- "individual": weights of the ANN that composed the individual
- "fitness_train": metric used to optimize the GA
- "y_pred_prob": probabilities with de ANN for train data
- "y_real": real y data for train
- "loss", "tp", "fp", "tn", "fn", "precision", "recall", "auc", "prc", "accuracy": metrics calculated for train data
- "fitness_test" same metric used for GA optimization, but with test data
- "loss_test", "tp_test", "fp_test", "tn_test", "fn_test", "precision_test", "recall_test", "auc_test", "prc_test", "accuracy_test": metrics obtained for test data after train

* *print_metric_evolution*: Function to print the metric desired evolution over the generations.

```python
model.print_metric_evolution()
model.print_metric_evolution(metric="accuracy", save = True)
```
  * metric: option to choose other metric than *fitness_train*.
  * save: by default is **False**. In case that **True** is selected, it will be saved in the folder created beforehand automatically.

* *save_bests_metrics*: Creates a *.csv* on the folder of results with the data of the bests individuals on each generation. Same data that the one obtained with the function *get_bests_metrics*.

```python
model.save_bests_metrics()
```