import numpy as np
import fnmatch
import re
from matplotlib import pyplot as plt
import seaborn as sns

from geneticalgorithm2 import geneticalgorithm2 as ga

from params import params as p
from params import ga_params as ga_p
from params import ann_params as ann_p

from callbacks import Callbacks
from fitness import *

import warnings
warnings.filterwarnings("ignore")

class GA_ANN():
    def __init__(self,
                ann_p = ann_p,
                ga_p = ga_p,
                var_type = p["var_type"],
                timeout = p["timeout"],
                fitness_function = fitness_function):

        data, X, y = LoadData().get_data()
        X_train, y_train, X_test, y_test = LoadData().get_model_data()

        # 1. Calculate varbound for our ann network
        network_len = X_train.shape[1]*ann_p["Input layer"]
        for index, item in enumerate(ann_p["Hiddens layers"]):
            if not index:
                network_len = network_len + (item*ann_p["Input layer"])
            else:
                network_len = network_len + (item*ann_p["Hiddens layers"][index-1])
        network_len = network_len + (ann_p["Output layer"]*ann_p["Hiddens layers"][-1])

        varbound=np.array([[-2,2]]*network_len, dtype=object)
        
        model = ga(function = fitness_function,
                   dimension = network_len,
                   variable_type = var_type,
                   variable_boundaries= varbound,
                   function_timeout = timeout,
                   algorithm_parameters=ga_p)

        self.model = model.run(callbacks=[Callbacks.SavePopulation_bests()], seed = None, no_plot = True)
        self.data = data
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_data(self):
        return self.data, self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test

    def get_result(self):
        return self.model

    def get_bests_metrics(self,
                          fold = "../results/" + p["input_name"] + "/tmp/"):
        # Joining npz
        bests_list = fnmatch.filter(os.listdir(fold), 'population_bests_' + '*')
        ## Order first generations: _1, _10 ... _2, _20 by _1, _2, _3...
        ### new index
        new_order = []
        for idx, item in enumerate(bests_list):
            end = item.rsplit('_', 1)[1]
            end = int(re.sub('\.npz$', '', end))
            new_order.append(end) 
        new_order = list(np.argsort(new_order))
        bests_list_ordered = []
        for item in new_order:
            bests_list_ordered.append(bests_list[item])

        bests_all = [np.load(fold+fname) for fname in bests_list_ordered]
        merged_bests = []
        for idx, data in enumerate(bests_all):
            if idx == 0:
                tmp = pd.DataFrame.from_dict({item: data[item] for item in data.files}, orient='index').T
                merged_bests = tmp
            else:
                tmp = pd.DataFrame.from_dict({item: data[item] for item in data.files}, orient='index').T
                merged_bests = merged_bests.append(tmp, ignore_index=True)

        np.savez(fold+'/population_bests.npz', **merged_bests)
        npz = np.load(fold+'/population_bests.npz', allow_pickle=True)
        df= pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index').T

        self.df = df

        return self.df

    def print_metric_evolution(self,
                               metric = "fitness_train",
                               save = False,
                               dim = (15,10),
                               fold = "../results/" + p["input_name"]):

        plt.figure(figsize=dim)
        ax = sns.lineplot(data=self.df, x="generation", y=metric)
        ax.set(xlabel="Generation", ylabel=metric)
        plt.title("Evolution " + metric)

        if save:
            plt.savefig(fold + '/' + metric + '.png')
        else:
            plt.show()

        return
        
    def save_bests_metrics(self,
                           fold = "../results/" + p["input_name"] + "/"):
        self.df.to_csv(fold + "bests.csv", header=True, index=False)
        return