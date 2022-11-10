# UTIL FUNCTIONS TO USE IN TRAIN
## -------------------------------------------------------------------##

import os

##1. Function to create new folder
def folder_create(folder):
    """
    Function to check if a folder exist, otherwise, create one named like indicated
    :params: folder - name of the new folder 
    :return: 
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


##2. Get params from JSON (crossover_type, keras optimizer...)
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