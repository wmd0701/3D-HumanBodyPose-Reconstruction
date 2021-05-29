import torch
from .linear_model import LinearModel
import numpy as np

class IterativeRegressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, iterations, initialization, batch_size):
        super(IterativeRegressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        smpl_mean_params = np.tile(initialization, (batch_size, 1))
        self.register_buffer('init_theta', torch.from_numpy(smpl_mean_params).float())
    '''
        param:
            inputs: features, e.g extracted from resnet50
        
        return:
            theta: 
    '''
    def forward(self, inputs):
        size = inputs.shape[0]
        theta = self.init_theta[:size]
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
        return theta