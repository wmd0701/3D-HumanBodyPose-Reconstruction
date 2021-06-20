from torch import nn
from .iterative_regressor import IterativeRegressor
from .util import load_smpl_mean_params

def get_regressor(name, batch_size):
    
    if name == 'iterative':
        fc_layers = [2048 + 82, 1024, 1024, 82]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        iterations = 3
        initialization = load_smpl_mean_params("data/smpl_mean_params_S1_S5_S6_S7_S8.npz")

        regressor = IterativeRegressor(
            fc_layers,
            use_dropout,
            drop_prob,
            use_ac_func,
            iterations,
            initialization,
            batch_size)
        
        return regressor, 82
    elif name == 'simple':
        regressor = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        )
        return regressor, 1024
    else:
        raise Exception(f'Regressor \'{name}\' is not supported.')

        

