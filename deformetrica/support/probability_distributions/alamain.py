import numpy as np
import torch
from torch.autograd import Variable
import scipy.spatial as sp

from ...support import utilities


class AlamainDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.cp = np.zeros((1,))
        self.covariance = 10
        self.covariance_orth = 20
        self.box = np.zeros([2,2])
        self.power = 1

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_variance(self, cov):
        self.covariance_orth = cov

    def set_cp(self, cp):
        self.cp = cp

    def set_dir(self, m):
        self.dir = np.zeros(m.shape)
        self.dir_orth = np.zeros(m.shape)
        if m.shape[-1] == 3:
            self.dir_orth2 = np.zeros(m.shape)
        for i in range(m.shape[0]):
            if m.any():
                self.dir[i] = m[i]/np.linalg.norm(m[i])
                self.dir_orth[i] = np.random.rand(m[i].size)
                self.dir_orth[i] -= np.dot(self.dir_orth[i], self.dir[i]) * self.dir[i]
                self.dir_orth[i] /= np.linalg.norm(self.dir_orth[i])
            else:
                self.dir[i] = np.random.rand(m[i].size)
                self.dir[i] /= self.dir[i]
                self.dir_orth[i] = np.random.rand(m[i].size)
                self.dir_orth[i] -= np.dot(self.dir_orth[i], self.dir[i]) * self.dir[i]
                self.dir_orth[i] /= np.linalg.norm(self.dir_orth[i])
            if m.shape[-1] == 3:
                self.dir_orth2[i] = np.cross(self.dir[i], self.dir_orth[i])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        res = 0
        if self.cp.shape[-1] == 2:
            for k in range(observation.shape[0]):
                if (self.box[:, 0] > observation[k]).any() or (self.box[:, 1] < observation[k]).any():
                    return - float('inf')
                for i in range(self.cp.shape[0]):
                    y = observation[k] - self.cp[i]
                    res -= self.power*np.exp(-np.dot(y, self.dir[i])**2/self.covariance - np.dot(y, self.dir_orth[i])**2/self.covariance_orth)
        else:
            for k in range(observation.shape[0]):
                if (self.box[:, 0] > observation[k]).any() or (self.box[:, 1] < observation[k]).any():
                    return - float('inf')
                for i in range(self.cp.shape[0]):
                    y = observation[k] - self.cp[i]
                    res -= self.power * np.exp(
                        -np.dot(y, self.dir[i]) ** 2 / self.covariance - np.dot(y, self.dir_orth[i]) ** 2 / self.covariance_orth - np.dot(y, self.dir_orth2[i]) ** 2 / self.covariance_orth)

        dist = sp.distance_matrix(observation, observation)
        res -= self.power/10 * np.sum(np.exp(-dist**2/(2*200)))/2


        return res

