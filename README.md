# Artifical Neural Network trained with Genetic Algorithm Model

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
python3.7.9 -m venv DopANAN/
source bin/activate
```

3. Upgrade pip and install project requirements 
```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project workflow

In params.py
First define params
Then ga_params
Finally ann_params

Then use run script 

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
* *inputfile*: here is where you must indicate where your dataset *.csv* file is located, starting your path from the folder **/src**.
* *input_name*: name to save the results of your model.
* *normalize_data*: in case your data is not normalized, the default option is to normalize it between [-1 1] due to be recommended for ANN models.
* *test_size*: proportion of the data to use as test after having trained with the *1-test_size* proportion part.
* *output_colname*: name of the columns which provides the output of the dataset.
* *fitness_metric*: metric used as **fitness** for the GA. Available: *loss*, *accuracy*, *auc*, *precision*, *recall* and *prc*.
* *var_type*: GA params before run it, indicates the kind of data will compose de genome of the individuals of the new population.
* *timeout*: another GA param before run it, time to wait until GA start running function. If the time exceeds this variable, it will raise an error.

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
* *max_num_iteration*: number of individuals on each generation.
* *population_size*: maximun number of generations.
* *mutation_probability*: proportion of mutation for the individuals creating the next generation.
* *elit_radio*: proportion of elitism used creating the next generation.
* *parents_portion*: proportion of the individuals for the next generations belonging to the parents pool.
* *crossover_type*: type of crossover to cross parents. Available: 'one_point', 'two_point', 'uniform', 'segment', 'shuffle'... More info in [Geneticalgorithm2](https://github.com/PasaOpasen/geneticalgorithm2/blob/master/geneticalgorithm2/crossovers.py) package.
* *max_iteration_without_improv*: stopping criteria value that indicates the number of generations to stop the evolution if the fitness value does not improve.

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
* *Activation Function*


### How to run

