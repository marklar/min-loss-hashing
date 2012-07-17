
# e.g.: Point = namedtuple('Point', ['x', 'y'], verbose=True)
from collections import namedtuple

# import hyperparams

# -- CONFIG --
n_bits = 64

def create_data():
    pass

def train(data):
    """  """
    
    
def run_trial(data, n_bits):
    """  """
    hyperparams = Hyperparams(data, n_bits)
    return train(hyperparams)

#
# What's the goal of each trial?
# Is it to find the best_params: rho, shink_w?
#    rho:      neighborliness threshold
#    shrink_w: weight decay
#
data = create_data()
n_trials = 3
for i in range(n_trials):
    run_trial(data)
