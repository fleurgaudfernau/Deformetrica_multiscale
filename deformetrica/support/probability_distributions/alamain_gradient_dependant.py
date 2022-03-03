import numpy as np
import torch
from torch.autograd import Variable
import scipy.spatial as sp

from ...support import utilities


class AlamainGradientDependantDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.image_grad = []
        self.alpha = 1
        self.var = 1
        self.box = []
        self.points = []
        self.module_directions = []
        self.module_variances = []
        self.module_intensities = []

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_variance(self, cov):
        self.var = cov

    def set_image_grad(self, grad):
        self.image_grad = grad

    def construct_sparse_matrix(self, module_centers):
        dim = self.points.shape[:-1]
        sparse_matrix = torch.zeros(dim).double()
        for k in range(self.module_directions.shape[0]):
            if not self.module_directions[k].detach().numpy().any():
                self.module_directions[k,0] = 1
            dir = self.module_directions[k,0]/torch.norm(self.module_directions[k,0])
            if dir.shape[0] == 2:
                e = torch.rand(dir.shape[0], dtype=torch.float64)
                e -= torch.dot(e,dir)*dir
                e /= torch.norm(e)
            else:
                e = self.module_directions[k,1]/torch.norm(self.module_directions[k,1])
                e -= torch.dot(e, dir) * dir
                e /= torch.norm(e)
                e2 = torch.cross(dir,e)

            y = torch.tensor(self.points - module_centers[k])
            if dir.shape[0] == 2:
                dist = (torch.mm(y.view(-1,2), dir.view(2,1))**2/self.module_variances[k,0]**2 + torch.mm(y.view(-1,2), e.view(2,1))**2/self.module_variances[k,1]**2).reshape(dim)
            else:
                dist = (torch.mm(y.view(-1,3), dir.view(3,1))**2/self.module_variances[k,0]**2 + torch.mm(y.view(-1,3), e.view(3,1))**2/self.module_variances[k,1]**2 + torch.mm(y.view(-1,3), e2.view(3,1))/self.module_variances[k,2]**2).reshape(dim[1:])

            sparse_matrix += torch.exp(-dist)*self.module_intensities[k]
        return sparse_matrix

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        sparse_matrix = np.array(self.construct_sparse_matrix(observation))
        for i in range(observation.shape[0]):
            if (self.box[:, 0] > observation[i]).any() or (self.box[:, 1] < observation[i]).any():
                return - float('inf')
        res = - self.alpha * np.sum(np.multiply(self.image_grad, sparse_matrix))

        dist = sp.distance_matrix(observation, observation)
        res -= self.alpha/100 * np.sum(np.exp(-dist**2/(2*200)))/2

        return res

