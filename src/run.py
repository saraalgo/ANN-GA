import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH + "/src/")

from ga_ann import *
from params import *

model = GA_ANN()

result = model.get_result()
bests = model.get_bests_metrics()

# To print metric evolution
model.print_metric_evolution()
model.print_metric_evolution(metric="accuracy", save = True)
# To save bests
model.save_bests_metrics()
