import numpy as np
import torch
from torch.autograd import Variable


class UniformDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, max=0, proba=0):
        self.max = max
        if proba == 0: self.proba = 1/max * np.ones(max)
        else: self.proba = proba

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_max(self, max):
        self.max = max

    def get_max(self):
        return self.max

    def set_probability(self,w):
        self.proba = w

    def get_variance_sqrt(self):
        return 1

    def set_variance_sqrt(self, std):
        1


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        return np.array(np.random.choice(self.max,p=self.proba))

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        return np.array(np.log(self.proba[observation]))

