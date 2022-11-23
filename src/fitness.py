import tensorflow as tf

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from params import params as p
from params import ann_params as ann_p

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

if ann_p["Output layer"] == 1 and ann_p["Loss"] == "binary_crossentropy":
    METRICS.append(tf.keras.metrics.BinaryAccuracy(name='accuracy'))
elif ann_p["Output layer"] == 1 and ann_p["Loss"] == "mse":
    METRICS.append(tf.keras.metrics.Accuracy(name='accuracy'))    
else:
    METRICS.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))

# Get params from dictionary (crossover_type, keras optimizer...)
def get_params(conf, element):
    """
    Function to get params from JSON to use str as function
    :params: conf - select json configuration
    :params: element - element from json from which we want the modifiable params
    :return: params - params of the element of that configuration
    """
    keywords = conf.get(element+" params", "")
    params = {}

    for key in keywords.split():
        params[key] = conf[element + " params " + key]
    
    return params

## Optimizers TF
params_optimizer_dop = get_params(ann_p, "Optimizer")
KerasOptimizers = {
    "rmsprop" : tf.keras.optimizers.RMSprop(**params_optimizer_dop),
    "sgd" : tf.keras.optimizers.SGD(**params_optimizer_dop)
}

## Load Data
class LoadData():
    def __init__(self,
                inputfile = p["inputfile"],
                output_colname = p["output_colname"],
                normalize = p["normalize_data"],
                test_size = p["test_size"]):

        # 1. Read data
        data = pd.read_csv(inputfile)
        X = data.drop(output_colname, axis = 1)
        y = data[output_colname]

        # 2. Normalize data
        if normalize:
            scaler = MinMaxScaler(feature_range=(-1,1))
            X_array = scaler.fit_transform(X)
        else:
            X_array = X.to_numpy()

        y_array = y.to_numpy()

        # 3. Divide data
        X_train, X_test, y_train, y_test = train_test_split(X_array, y_array,
                                                            test_size = test_size)

        self.data = data
        self.X = X_array
        self.y = y_array
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_data(self):
        return self.data, self.X, self.y

    def get_model_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test


## Get Ann Model
class AnnModel():
    def __init__(self,
                X,
                ann_p = ann_p,
                metrics=METRICS,
                optimizer = KerasOptimizers):
            """
            Init class to create the architecture of model of the ANN with TF
            :params: X - vector with random weights for the ANN
            :params: ann_p - ann_puration to indicate ANN params 
            :params: idim - number of features of the input data 
            :return: annmodel - initiate Dopamine ANN model in TF for DAM, DM or Control mode
            """
            data = LoadData()
            X_train, y_train, X_test, y_test = data.get_model_data()
            idim = X_train.shape[1]

            annmodel = tf.keras.Sequential()
            annmodel.add(tf.keras.Input(shape=(idim,)))
            annmodel.add(tf.keras.layers.Dense(ann_p["Input layer"], input_dim = idim, activation=ann_p["Activation Function"], kernel_initializer = tf.constant_initializer(X[:(ann_p["Input layer"]*idim)])))
            counter = ann_p["Input layer"]*idim
            for index, item in enumerate(ann_p["Hiddens layers"]):
                if not index:
                    annmodel.add(tf.keras.layers.Dense(item, activation=ann_p["Activation Function"], kernel_initializer = tf.constant_initializer(X[counter:(counter+(item*ann_p["Input layer"]))])))
                    counter = counter + (item*ann_p["Input layer"])
                else:
                    annmodel.add(tf.keras.layers.Dense(item, activation=ann_p["Activation Function"], kernel_initializer = tf.constant_initializer(X[counter:(counter+(item*ann_p["Hiddens layers"][index-1]))])))
                    counter = counter + (item*ann_p["Hiddens layers"][index-1])
            
            if ann_p["Output layer"] == 1:
                annmodel.add(tf.keras.layers.Dense(ann_p["Output layer"], activation="sigmoid", kernel_initializer = tf.constant_initializer(X[counter:])))
            else:
                annmodel.add(tf.keras.layers.Dense(ann_p["Output layer"], activation="softmax", kernel_initializer = tf.constant_initializer(X[counter:])))
            
            annmodel.compile(loss=ann_p["Loss"], optimizer=optimizer, metrics=metrics) 
            self.annmodel = annmodel
            self.X = X

    
    def predict(self, X_train, y_train, metric = p["fitness_metric"], save="../results/" + p["input_name"]):
        yPred = self.annmodel.evaluate(X_train, y_train,
                              batch_size=ann_p["Batch size"], verbose=ann_p["Verbose"])
        yPred.insert(0, y_train.tolist())
        y_pred_prob = self.annmodel.predict(X_train, 
                                   batch_size=ann_p["Batch size"], verbose=ann_p["Verbose"])
        yPred.insert(0, y_pred_prob.ravel())
        names = ['y_pred_prob','y_real']+self.annmodel.metrics_names

        # Save metrics of only best individual in each generation
        if not os.path.exists(save):
            os.makedirs(save)
            os.makedirs(save+'/tmp')
        file_metrics = Path(save + '/tmp/metrics_tmp_best_population.npz')
        file_metrics.touch(exist_ok=True)
        if os.stat(file_metrics).st_size == 0:
            np.savez(file_metrics, names, yPred)
        else:
            npz = np.load(file_metrics, allow_pickle=True)
            old_metrics = list(npz.f.arr_1)
            if metric in ["loss", "fn", "fp"]:
                new_metrics = old_metrics if old_metrics[names.index(metric)] < yPred[names.index(metric)] else yPred
                np.savez(file_metrics, names, new_metrics)
            else:
                new_metrics = old_metrics if old_metrics[names.index(metric)] > yPred[names.index(metric)] else yPred
                np.savez(file_metrics, names,new_metrics)
            
        self.yPred = yPred
        return self.yPred[names.index(metric)]

    def predict_test(self, X_test, y_test, metric = p["fitness_metric"]):
        yPred = self.annmodel.evaluate(X_test, y_test,
                            batch_size=ann_p["Batch size"], verbose=ann_p["Verbose"])

        names = self.annmodel.metrics_names
        yPred.insert(0, yPred[names.index(metric)])
            
        self.yPred = yPred
        return self.yPred

## Fitness function

def fitness_function(X):
    # initiate dopamine network 
    X_train, y_train, X_test, y_test = LoadData().get_model_data()
    annmodel = AnnModel(X)
    fitness = annmodel.predict(X_train, y_train)
    if p["fitness_metric"] in ["loss", "fn", "fp"]:
        pass
    else:
        fitness = -fitness
    return fitness