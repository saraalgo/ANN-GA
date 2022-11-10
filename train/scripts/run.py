## Set globals
DATASET = "Audit"

ANN_TYPE = "class"
CLASS_TYPE = "binary" # only in case class is selected in ANN_type


# NOT MODIFY FROM HERE BELOW!!!!!!
#================================================================================================
#================================================================================================


### PACKAGES
#=================================================================

import os
## Change the current working directory to principal git path
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split("ANN-GA",1)[0] + "ANN-GA/")
from utils.ANN_GA import *
from utils.utils import *

## Basic packages
import numpy as np
import json

## Training packages
import tensorflow as tf
from geneticalgorithm2 import geneticalgorithm2 as ga 
from geneticalgorithm2 import Callbacks, Crossover

## Set seed to reproduce always the same results
from numpy.random import seed
seed(123)
tf.random.set_seed(123)

### GLOBALS
#=================================================================
DATA_PATH = "data/" + DATASET + "/"
folder_create("train/results/" + DATASET)

## Load .json from globals
if ANN_TYPE == "class":
    CONFIG_ANN = "config/ANN_classification.json"
else:
    CONFIG_ANN = "config/ANN_regression.json"

f = open(CONFIG_ANN,)
config_ann = json.load(f)

f = open("config/ANN_architecture.json",)
arch_ann = json.load(f)

f = open("config/GA_params.json",)
ga_params = json.load(f)


## Optimizers TF

params_optimizer_dop = get_params(config_ann, "Optimizer")
KerasOptimizers = {
    "rmsprop" : tf.keras.optimizers.RMSprop(**params_optimizer_dop),
    "sgd" : tf.keras.optimizers.SGD(**params_optimizer_dop)
}


### LOAD DATA
#=================================================================

with np.load(DATA_PATH +"train_test.npz") as data:
    train_idata = data["X_train"]
    train_odata = data["y_train"]
    print("Reading Train Data")
    test_idata = data["X_test"]
    test_odata = data["y_test"]
    print("Reading Test Data")


