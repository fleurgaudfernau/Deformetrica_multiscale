import numpy as np
import torch
from torch.autograd import Variable
import scipy.spatial as sp

from ...support import utilities


class AlamainGradientAllInDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.image_grad = []
        self.alpha = 1
        self.var = 1
        self.box = []
        self.points = []
        self.dimension = 2
        self.subject = 0
        self.iter = 0

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_variance(self, cov):
        self.var = cov

    def set_image_grad(self, grad):
        self.image_grad = grad

    def construct_sparse_matrix(self, module_positions, module_intensity, module_variances, module_directions):
        dim = self.points.shape[:-1]
        sparse_matrix = torch.zeros(dim).double()
        for k in range(module_directions.shape[0]):
            if not module_directions[k].any():
                module_directions[k, 0] = 1
            dir = torch.tensor(module_directions[0] / np.linalg.norm(module_directions[0]))
            if dir.shape[0] == 2:
                e = torch.rand(dir.shape[0], dtype=torch.float64)
                e -= torch.dot(e, dir) * dir
                e /= torch.norm(e)
            else:
                e = torch.tensor(module_directions[1] / np.linalg.norm(module_directions[1]))
                e -= torch.dot(e, dir) * dir
                e /= torch.norm(e)
                e2 = torch.cross(dir, e)

            y = torch.tensor(self.points - module_positions[k])
            if dir.shape[0] == 2:
                dist = (torch.mm(y.view(-1, 2), dir.view(2, 1)) ** 2 / module_variances[0] ** 2 + torch.mm(
                    y.view(-1, 2), e.view(2, 1)) ** 2 / module_variances[1] ** 2).reshape(dim)
            else:
                dist = (torch.mm(y.view(-1, 3), dir.view(3, 1)) ** 2 / module_variances[0] ** 2 + torch.mm(
                    y.view(-1, 3), e.view(3, 1)) ** 2 / module_variances[1] ** 2 + torch.mm(y.view(-1, 3),
                                                                                                    e2.view(3, 1))**2 /
                        module_variances[2] ** 2).reshape(dim)

            sparse_matrix += torch.exp(-dist) * module_intensity
        return sparse_matrix

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        module_positions = observation[:self.dimension]
        module_intensity = observation[self.dimension]
        module_variances = observation[self.dimension+1:2*self.dimension+1]
        module_directions = observation[2*self.dimension+1:].reshape([self.dimension-1, self.dimension])

        #if module_intensity > 0: return -float('inf')

        sparse_matrix = np.array(self.construct_sparse_matrix(module_positions, module_intensity, module_variances, module_directions))

        #if module_intensity > 0: print('intensity' + str(module_intensity))

        if (self.box[:, 0] > module_positions).any() or (self.box[:, 1] < module_positions).any() or module_intensity < 0:
            return - float('inf')


        #res = - self.alpha * np.abs(np.sum(np.multiply(self.image_grad[self.subject], sparse_matrix)))/100000000000
        res = - np.linalg.norm(self.image_grad[self.subject],2)
        #if res != 0: print(res)
        #print('regu due to sparse: ' + str(res))
        #dist = sp.distance_matrix(observation[:self.dimension], observation[:self.dimension])
        #res -= self.alpha / 100 * np.sum(np.exp(-dist ** 2 / (2 * 200))) / 2

        res -= np.linalg.norm(module_intensity.ravel() - self.mean_int,ord=1)/self.scale_int
        #print('+ regu due to intensity: ' + str(res))

        res -= 0.5 * np.sum((module_variances.ravel() - self.mean_var) ** 2)/self.variance_var
        #print('+ regu due to variances: ' + str(res))

        res -= 0.5 * np.sum((module_directions.ravel() - self.mean_dir) ** 2)/self.variance_dir
        #print('+regu due to directions: ' + str(res))

        self.iter += 1
        self.subject = np.mod(np.int(self.iter/2), self.number_of_subjects)

        return res
