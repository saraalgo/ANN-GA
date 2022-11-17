
params = {
    "inputfile": "../data/data.csv",

    "fitness_metric": "loss",
    "var_type": "real",
    "timeout": 5000
}

ga_params = {
    "max_num_iteration": 2,
    "population_size": 7,
    "mutation_probability": 0.1, 
    "elit_ratio": 0.01,
    "parents_portion": 0.3,
    "crossover_type": "uniform",
    "max_iteration_without_improv": 50  
}

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
