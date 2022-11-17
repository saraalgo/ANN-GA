import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga

from params import params as p
from params import ga_params as ga_p
from params import ann_params as ann_p

from fitness import *

class Pipeline():
    def __init__(self,
                inputfile,
                output_colname = "output",
                normalize = False,
                test_size = 0.3,
                ann_p = ann_p,
                ga_p = ga_p,
                var_type = p["var_type"],
                timeout = p["timeout"],
                fitness_function = fitness_function):

        X_train, y_train, X_test, y_test = LoadData(inputfile,
                                                    output_colname,
                                                    normalize,
                                                    test_size).get_model_data()

        # 1. Calculate varbound for our ann network
        network_len = X_train.shape[1]*ann_p["Input layer"]
        for index, item in enumerate(ann_p["Hiddens layers"]):
            if not index:
                network_len = network_len + (item*ann_p["Input layer"])
            else:
                network_len = network_len + (item*ann_p["Hiddens layers"][index-1])
        network_len = network_len + (ann_p["Output layer"]*ann_p["Hiddens layers"][-1])

        varbound=np.array([[-2,2]]*network_len)
        
        model = ga(function = fitness_function,
                   dimension = network_len,
                   variable_type = var_type,
                   variable_boundaries= varbound,
                   function_timeout = timeout,
                   algorithm_parameters=ga_p)

        self.model = model.run()
        return self.model

