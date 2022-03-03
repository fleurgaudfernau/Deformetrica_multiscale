import numpy as np
import torch
from torch.autograd import Variable
import scipy.spatial as sp

from ...support import utilities


class AlamainGradientDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.image_grad = []
        self.alpha = 1
        self.var = 1
        self.box = []

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_variance(self, cov):
        self.var = cov

    def set_image_grad(self, grad):
        self.image_grad = grad

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        res = 0
        for i in range(observation.shape[0]):
            if (self.box[:, 0] > observation[i]).any() or (self.box[:, 1] < observation[i]).any():
                return - float('inf')
            else:
                res -= self.alpha*self.image_grad[int(observation[i][0]),int(observation[i][1])]

        dist = sp.distance_matrix(observation, observation)
        res -= self.alpha/10 * np.sum(np.exp(-dist**2/(2*200)))/2

        return res

