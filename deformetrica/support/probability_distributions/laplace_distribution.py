import numpy as np
import torch
from torch.autograd import Variable

from ...support import utilities


class LaplaceDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.mean = np.zeros((1,))
        self.scale = 0.01

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_scale(self, alpha):
        self.scale = alpha

    def set_mean(self, m):
        self.mean = m

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        return np.random.laplace(0,self.scale,self.mean.shape())

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        if np.max(observation) > 0.0:
           return -float('inf')
        else:
            return -np.linalg.norm(observation.ravel() - self.mean.ravel(),ord=1)/self.scale

    def compute_log_likelihood_torch(self, observation, tensor_scalar_type, device='cpu'):
        """
        Torch inputs / outputs.
        Returns only the part that includes the observation argument.
        """
        observation = utilities.move_data(observation, dtype=tensor_scalar_type, device=device)
        return - torch.norm(observation,p=1)/self.scale
