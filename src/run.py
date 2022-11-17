import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH + "/src/")

from ga_ann import *
from params import *

locals().update(params)
locals().update(ga_params)
locals().update(ann_params)

model = Pipeline(inputfile = p["inputfile"],
                output_colname="output",
                normalize=True,
                test_size=0.2,
                ann_p = ann_p,
                ga_p = ga_p,
                var_type = p["var_type"],
                timeout = p["timeout"],
                fitness_function = fitness_function)

