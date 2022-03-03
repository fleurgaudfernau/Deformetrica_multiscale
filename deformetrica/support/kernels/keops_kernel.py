import torch
import numpy as np

from ...support.kernels import AbstractKernel
from ...core import default, GpuMode
from pykeops.torch import Genred


import logging
logger = logging.getLogger(__name__)


class KeopsKernel(AbstractKernel):
    def __init__(self, gpu_mode=default.gpu_mode, kernel_width=None, cuda_type=None, **kwargs):
        super().__init__('keops', gpu_mode, kernel_width)
        
        if cuda_type is None:
            cuda_type = default.dtype

        self.cuda_type = cuda_type

        #ajout fg
        import numpy as np
        if not isinstance(self.kernel_width, np.ndarray):
            self.gamma = 1. / default.tensor_scalar_type([self.kernel_width ** 2])

            self.gaussian_convolve = []
            self.point_cloud_convolve = []
            self.varifold_convolve = []
            self.gaussian_convolve_gradient_x = []
            
            #Genred: creates a new generic operation
            for dimension in [2, 3]:
                
                self.gaussian_convolve.append(Genred(
                    "Exp(-G*SqDist(X,Y)) * P",
                    ["G = Pm(1)", #1st input: no indexation, the input tensor is a vector of dim 1 and not a 2d array.
                    "X = Vi(" + str(dimension) + ")",  # 2nd input: dimension (2/3) vector per line (i means axis 0)
                    "Y = Vj(" + str(dimension) + ")", # 3rd input: dimension (2/3) vector per column (j means axis 1)
                    "P = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))
                
                self.point_cloud_convolve.append(Genred(
                    "Exp(-G*SqDist(X,Y)) * P",
                    ["G = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "P = Vj(1)"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                self.varifold_convolve.append(Genred(
                    "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                    ["G = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Nx = Vi(" + str(dimension) + ")",
                    "Ny = Vj(" + str(dimension) + ")",
                    "P = Vj(1)"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                self.gaussian_convolve_gradient_x.append(Genred(
                    "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                    ["G = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Px = Vi(" + str(dimension) + ")",
                    "Py = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

        #ajout fg
        if isinstance(self.kernel_width, np.ndarray):
            self.gamma = torch.from_numpy(1. /self.kernel_width**2)
            #print("self.gamma", self.gamma)

            self.gaussian_convolve = []
            self.gaussian_convolve_sum = []
            self.point_cloud_convolve = []
            self.varifold_convolve = []
            self.gaussian_convolve_gradient_x = []
            self.gaussian_convolve_gradient_x2 = []
            self.gaussian_convolve_gradient_x_sum = []

            for dimension in [2, 3]:
                #ajout de sigma² dans chaque axe -> pourrait être anisotrope
                
                self.gaussian_convolve.append(Genred(
                    "Exp(-G*SqDist(X,Y)) * P",
                    ["G = Vj(1)", 
                    "X = Vi(" + str(dimension) + ")",  # 2nd input: dimension (2/3) vector per line (i means axis 0)
                    "Y = Vj(" + str(dimension) + ")", # 3rd input: dimension (2/3) vector per column (j means axis 1)
                    "P = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type)) #SUM = 1: sum over j

                #kernel adapted to different sigmas
                self.gaussian_convolve_gradient_x.append(Genred(
                    "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                    ["G = Vj(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Px = Vi(" + str(dimension) + ")",
                    "Py = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))
                
                #kernel better adapted to different sigmas
                self.gaussian_convolve_gradient_x2.append(Genred(
                    "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y) * G",
                    ["G = Vj(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Px = Vi(" + str(dimension) + ")",
                    "Py = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                #modif fg: somme de kernels
                self.I = torch.tensor([torch.min(self.gamma)]) #grand sigma
                self.J =torch.tensor([torch.max(self.gamma)]) #petit sigma
                self.K =torch.tensor([(torch.min(self.gamma)+torch.max(self.gamma))/2])
                #poids
                self.V = torch.tensor([1/2])
                self.W = torch.tensor([1/2])
                self.Z = torch.tensor([1/3])
                
                #self.J = torch.max(self.gamma)
                #self.K = (self.I + self.J) / 2
                #print("self.gamma[0:5]", self.gamma[0:5])
                #print("self.I, self.J, self.K", self.I, self.J, self.K)
                #I, J, K tensor(0.0204, device='cuda:0') tensor(1., device='cuda:0') tensor(0.5102, device='cuda:0')


                self.gaussian_convolve_sum.append(Genred(
                    "(V*Exp(-I*SqDist(X,Y)) + W*Exp(-J*SqDist(X,Y))) * P",
                    ["I = Pm(1)", #1st input: no indexation, the input tensor is a vector of dim 1 and not a 2d array.
                    "X = Vi(" + str(dimension) + ")",  # 2nd input: dimension (2/3) vector per line (i means axis 0)
                    "Y = Vj(" + str(dimension) + ")", # 3rd input: dimension (2/3) vector per column (j means axis 1)
                    "J = Pm(1)",
                    #"K = Pm(1)",
                    "V = Pm(1)",
                    "W = Pm(1)",
                    #"Z = Pm(1)",
                    "P = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                self.gaussian_convolve_gradient_x_sum.append(Genred(
                    "(Px|Py) * (I *V* Exp(-I*SqDist(X,Y)) + J *W* Exp(-J*SqDist(X,Y))) * (X-Y)",
                    ["I = Pm(1)",
                    "J = Pm(1)",
                    #"K = Pm(1)",
                    "V = Pm(1)",
                    "W = Pm(1)",
                    #"Z = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Px = Vi(" + str(dimension) + ")",
                    "Py = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                self.point_cloud_convolve.append(Genred(
                    "Exp(-G*SqDist(X,Y)) * P",
                    ["G = Vj(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "P = Vj(1)"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                self.varifold_convolve.append(Genred(
                    "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                    ["G = Vj(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Nx = Vi(" + str(dimension) + ")",
                    "Ny = Vj(" + str(dimension) + ")",
                    "P = Vj(1)"],
                    reduction_op='Sum', axis=1, cuda_type=cuda_type))

                

                

    def __eq__(self, other):
        return AbstractKernel.__eq__(self, other) and self.cuda_type == other.cuda_type

    def convolve(self, x, y, p, mode='gaussian'):
        if mode == 'gaussian':
            #print("---------------------gaussian_convolve---------------------")
            assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
            assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
            assert isinstance(p, torch.Tensor), 'p variable must be a torch Tensor'

            # move tensors with respect to gpu_mode
            x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            #print("x", x.shape, "y", y.shape, "p", p.shape, "g", gamma.shape)
            if gamma.shape[0] > 1:
                
                res = self.gaussian_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), 
                                                device_id=device_id)
                #print("convolve res v1 [0]", res[0])
                """
                I = self.I.to(x.device, dtype=x.dtype)
                J = self.J.to(x.device, dtype=x.dtype)
                K = self.K.to(x.device, dtype=x.dtype)
                V = self.V.to(x.device, dtype=x.dtype)
                W = self.W.to(x.device, dtype=x.dtype)
                Z = self.Z.to(x.device, dtype=x.dtype)
                
                                
                res = self.gaussian_convolve_sum[d - 2](I, x.contiguous(), y.contiguous(), 
                                                        J, #K, 
                                                        V, W, #Z,
                                                        p.contiguous(), 
                                                        device_id=device_id)
                #print("convolve res sum [0]", res[0])
                #print()"""
            else:
                res = self.gaussian_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), 
                                                device_id=device_id)
            #print("res", res.shape)
            #x: nb points controle x dim, y : nb points ctrl x dim -> les positions des pt ctl
            #ou bien x 156 000*3 et y tjrs les pts controles (x position des pixels)

            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == 'pointcloud':
            assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
            assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
            assert isinstance(p, torch.Tensor), 'p variable must be a torch Tensor'

            # move tensors with respect to gpu_mode
            x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.point_cloud_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), device_id=device_id)
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == 'varifold':
            assert isinstance(x, tuple), 'x must be a tuple'
            assert len(x) == 2, 'tuple length must be 2'
            assert isinstance(y, tuple), 'y must be a tuple'
            assert len(y) == 2, 'tuple length must be 2'

            # tuples are immutable, mutability is needed to mode to device
            x = list(x)
            y = list(y)

            # move tensors with respect to gpu_mode
            x[0], x[1], y[0], y[1], p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x[0], x[1], y[0], y[1], p])
            assert x[0].device == y[0].device == p.device, 'x, y and p must be on the same device'
            assert x[1].device == y[1].device == p.device, 'x, y and p must be on the same device'

            x, nx = x
            y, ny = y
            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.varifold_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), nx.contiguous(), ny.contiguous(), p.contiguous(), device_id=device_id)
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):
        #print("---------------------convolve_gradient---------------------")
        
        if y is None:
            y = x
        if py is None:
            py = px

        assert isinstance(px, torch.Tensor), 'px variable must be a torch Tensor'
        assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
        assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
        assert isinstance(py, torch.Tensor), 'py variable must be a torch Tensor'

        # move tensors with respect to gpu_mode
        x, px, y, py = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, px, y, py])
        assert px.device == x.device == y.device == py.device, 'tensors must be on the same device'

        d = x.size(1)
        gamma = self.gamma.to(x.device, dtype=x.dtype)
        #print(gamma.shape[10])

        device_id = x.device.index if x.device.index is not None else -1
        if gamma.shape[0] > 1: #ajout fg
            #old version
            #res = (-2 * gamma[0] * self.gaussian_convolve_gradient_x[d - 2](gamma, x, y, px, py, device_id=device_id))
            #print("gradient res v1", res[0])

            #better adapted convolve_gradient
            res = (-2 * self.gaussian_convolve_gradient_x2[d - 2](gamma, x, y, px, py, device_id=device_id))
            #print("gradient res v2", res[0])

            #sum of kernels
            """
            I = self.I.to(x.device, dtype=x.dtype)
            J = self.J.to(x.device, dtype=x.dtype)
            K = self.K.to(x.device, dtype=x.dtype)
            V = self.V.to(x.device, dtype=x.dtype)
            W = self.W.to(x.device, dtype=x.dtype)
            Z = self.Z.to(x.device, dtype=x.dtype)
            res = (-2 * self.gaussian_convolve_gradient_x_sum[d - 2](I, J, V, W, #K, Z, 
                                                                    x, y, px, py, device_id=device_id))
            #print("gradient res sum", res[0])
            #print()"""
            
        else:
            res = (-2 * gamma * self.gaussian_convolve_gradient_x[d - 2](gamma, x, y, px, py, device_id=device_id))
        
        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res
