import logging
import math
import threading
import time

import torch

#import support.kernels as kernel_factory
from ...support.kernels import factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
#import support.utilities as utilities
from ...support.utilities import move_data, get_best_device
from .abstract_statistical_model import process_initial_data
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# def _subject_attachment_and_regularity(arg):
#     """
#     Auxiliary function for multithreading (cannot be a class method).
#     """
#     from .abstract_statistical_model import process_initial_data
#     if process_initial_data is None:
#         raise RuntimeError('process_initial_data is not set !')
#
#     # Read arguments.
#     freeze_sparse_matrix = False
#     (deformable_objects, multi_object_attachment, objects_noise_variance,
#      freeze_template, freeze_control_points, freeze_momenta,
#      exponential, sobolev_kernel, use_sobolev_gradient, tensor_scalar_type, gpu_mode) = process_initial_data
#     (i, template, template_data, control_points, momenta, with_grad, momenta_t, sparse_matrix, alpha) = arg
#
#     # start = time.perf_counter()
#     device, device_id = get_best_device(gpu_mode=gpu_mode)
#     # device, device_id = ('cpu', -1)
#     if device_id >= 0:
#         torch.cuda.set_device(device_id)
#
#     # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
#     template_data = {key: move_data(value, device=device, dtype=tensor_scalar_type,
#                                               requires_grad=with_grad and not freeze_template)
#                      for key, value in template_data.items()}
#     template_points = {key: move_data(value, device=device, dtype=tensor_scalar_type,
#                                                 requires_grad=with_grad and not freeze_template)
#                        for key, value in template.get_points().items()}
#     control_points = move_data(control_points, device=device, dtype=tensor_scalar_type,
#                                          requires_grad=with_grad and not freeze_control_points)
#     momenta = move_data(momenta, device=device, dtype=tensor_scalar_type,
#                                   requires_grad=with_grad and not freeze_momenta)
#
#     assert torch.device(
#         device) == control_points.device == momenta.device, 'control_points and momenta tensors must be on the same device. ' \
#                                                             'device=' + device + \
#                                                             ', control_points.device=' + str(control_points.device) + \
#                                                             ', momenta.device=' + str(momenta.device)
#
#     attachment, regularity = DeterministicAtlasHypertemplate._deform_and_compute_attachment_and_regularity(
#         exponential, template_points, control_points, momenta,
#         template, template_data, multi_object_attachment,
#         deformable_objects[i], objects_noise_variance, alpha,
#         device)
#
#     res = DeterministicAtlasHypertemplate._compute_gradients(
#         attachment, regularity,
#         freeze_template, momenta_t,
#         freeze_control_points, control_points,
#         freeze_momenta, momenta, freeze_sparse_matrix, sparse_matrix,
#         with_grad)
#     # elapsed = time.perf_counter() - start
#     # logger.info('pid=' + str(os.getpid()) + ', ' + torch.multiprocessing.current_process().name +
#     #       ', device=' + device + ', elapsed=' + str(elapsed))
#     return i, res


class DeterministicAtlasWithModule(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, number_of_subjects,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 gpu_mode=default.gpu_mode,
                 process_per_gpu=default.process_per_gpu,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='DeterministicAtlas', number_of_processes=number_of_processes,
                                          gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['hypertemplate_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None
        self.fixed_effects['momenta_t'] = None
        self.fixed_effects['module_intensities'] = None
        self.fixed_effects['module_positions'] = None
        self.fixed_effects['module_variances'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points
        self.freeze_momenta = freeze_momenta
        self.freeze_sparse_matrix = False

        self.alpha = 1

        # Deformation.
        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=factory(deformation_kernel_type,
                                          gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        self.exponential_t = Exponential(
            dense_mode=dense_mode,
            kernel=factory(deformation_kernel_type,
                                          gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications,
                                                                                               self.dimension)

        self.template = DeformableMultiObject(object_list)
        self.hypertemplate = DeformableMultiObject(object_list)
        # self.template.update()

        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = factory(deformation_kernel_type,
                                                         gpu_mode=gpu_mode,
                                                         kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()
        self.fixed_effects['hypertemplate_data'] = self.hypertemplate.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
            initial_momenta, self.number_of_control_points, self.dimension, number_of_subjects)
        self.fixed_effects['momenta'] = 0.0001*np.ones(self.fixed_effects['momenta'].shape)
        self.number_of_subjects = number_of_subjects

        self.fixed_effects['momenta_t'] = initialize_momenta(
            None, self.number_of_control_points, self.dimension, 1)

        # initial_cp = initialize_control_points(None, self.template, 40, None, self.dimension, False)
        # self.nb_modules = initial_cp.shape[0]
        # self.fixed_effects['module_positions'] = np.array([initial_cp,]*self.number_of_subjects)
        #self.fixed_effects['module_positions'] = np.array([[np.array(self.template.get_points()['image_points'].shape[:-1])/2]*initial_cp.shape[0]]*self.number_of_subjects)
        for i in range(object_list[0].bounding_box.shape[0]):
            object_list[0].bounding_box[i,0] = object_list[0].bounding_box[i,1]/2 - 10/2
            object_list[0].bounding_box[i, 1] = object_list[0].bounding_box[i, 1] / 2 + 10 / 2
        t = DeformableMultiObject(object_list)
        initial_cp = initialize_control_points(None, t, 3, None, self.dimension, False)
        self.nb_modules = initial_cp.shape[0]
        self.fixed_effects['module_positions'] = np.array([initial_cp, ] * self.number_of_subjects)

        # k = 0
        # j = 0
        # add = 1
        # while k < self.nb_modules - 1:
        #     self.fixed_effects['module_positions'][:,k,j] += add
        #     self.fixed_effects['module_positions'][:, k+1, j] -= add
        #     k += 2
        #     if j == self.fixed_effects['module_positions'].shape[2] - 1:
        #         j = 0
        #         add += 1
        #     else:
        #         j+=1

        self.fixed_effects['module_intensities'] = 0*np.ones([self.number_of_subjects, self.nb_modules])
        self.fixed_effects['module_variances'] = 5*np.ones([self.number_of_subjects, self.nb_modules, self.dimension])
        self.process_per_gpu = process_per_gpu

        self.regu_var_m = 5
        self.regu_var_m_ortho = 10

    def initialize_noise_variance(self, dataset, device='cpu'):
        if np.min(self.objects_noise_variance) < 0:
            hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta, momenta_t \
                = self._fixed_effects_to_torch_tensors(False, device=device)
            targets = dataset.deformable_objects
            targets = [target[0] for target in targets]

            residuals_torch = []
            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_control_points(control_points)
            for i, target in enumerate(targets):
                self.exponential.set_initial_momenta(momenta[i])
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals_torch.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, target))

            residuals = np.zeros((self.number_of_objects,))
            for i in range(len(residuals_torch)):
                residuals += residuals_torch[i].detach().cpu().numpy()

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k, obj in enumerate(self.objects_name):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(self.number_of_subjects)
                    self.objects_noise_variance[k] = nv
                    logger.info('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    def get_hypertemplate_data(self):
        return self.fixed_effects['hypertemplate_data']

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp
        # self.number_of_control_points = len(cp)

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['momenta']

    def set_momenta(self, mom):
        self.fixed_effects['momenta'] = mom

    def get_momenta_t(self):
        return self.fixed_effects['momenta_t']

    def set_momenta_t(self, mom):
        self.fixed_effects['momenta_t'] = mom

    def set_module_intensities(self, w):
        self.fixed_effects['module_intensities'] = w

    def get_module_intensities(self):
        return self.fixed_effects['module_intensities']

    def set_module_positions(self, c):
        self.fixed_effects['module_positions'] = c

    def get_module_positions(self):
        return self.fixed_effects['module_positions']

    def set_module_variances(self, sigma):
        self.fixed_effects['module_variances'] = sigma

    def get_module_variances(self):
        return self.fixed_effects['module_variances']

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            out['momenta_t'] = self.fixed_effects['momenta_t']
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        if not self.freeze_momenta:
            out['momenta'] = self.fixed_effects['momenta']
        if not self.freeze_sparse_matrix:
            out['module_positions'] = self.fixed_effects['module_positions']
            out['module_intensities'] = self.fixed_effects['module_intensities']
            out['module_variances'] = self.fixed_effects['module_variances']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            device, _ = get_best_device(self.gpu_mode)
            hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta, momenta_t, module_intensities, module_positions, module_variances \
                = self._fixed_effects_to_torch_tensors(False, device=device)
            self.exponential_t.set_initial_template_points(hypertemplate_points)
            self.exponential_t.set_initial_control_points(control_points)
            self.exponential_t.set_initial_momenta(momenta_t[0])
            self.exponential_t.move_data_to_(device=device)
            self.exponential_t.update()
            template_points = self.exponential_t.get_template_points()
            template_data = self.hypertemplate.get_deformed_data(template_points, hypertemplate_data)
            template_data = {key: value.detach().cpu().numpy() for key, value in template_data.items()}
            self.set_momenta_t(fixed_effects['momenta_t'])
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        if not self.freeze_momenta:
            self.set_momenta(fixed_effects['momenta'])
        if not self.freeze_sparse_matrix:
            self.set_module_positions(fixed_effects['module_positions'])
            self.set_module_variances((fixed_effects['module_variances']))
            self.set_module_intensities(fixed_effects['module_intensities'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def setup_multiprocess_pool(self, dataset):
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.deformable_objects],
                                                self.multi_object_attachment,
                                                self.objects_noise_variance,
                                                self.freeze_template, self.freeze_control_points, self.freeze_momenta,
                                                self.exponential, self.sobolev_kernel, self.use_sobolev_gradient,
                                                self.tensor_scalar_type, self.gpu_mode))

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        if self.number_of_processes > 1:
            targets = [target[0] for target in dataset.deformable_objects]
            (deformable_objects, multi_object_attachment, objects_noise_variance,
             freeze_template, freeze_control_points, freeze_momenta,
            exponential, sobolev_kernel, use_sobolev_gradient, tensor_scalar_type, gpu_mode) = process_initial_data
            device, device_id = get_best_device(gpu_mode=gpu_mode)

            self.exponential_t.set_initial_template_points(self.hypertemplate.get_points())
            self.exponential_t.set_initial_control_points(self.fixed_effects['control_points'])
            self.exponential_t.set_initial_momenta(self.fixed_effects['momenta_t'])
            self.exponential_t.move_data_to_(device=device)
            self.exponential_t.update()
            template_points = self.exponential_t.get_template_points()
            template_data = self.hypertemplate.get_deformed_data(template_points, self.fixed_effects['hypertemplate_data'])

            args = [(i, self.template,
                     template_data,
                     self.fixed_effects['control_points'],
                     self.fixed_effects['momenta'][i],
                     with_grad) for i in range(len(targets))]

            start = time.perf_counter()

            results = self.pool.map(_subject_attachment_and_regularity, args, chunksize=1)  # TODO: optimized chunk size
            # results = self.pool.imap_unordered(_subject_attachment_and_regularity, args, chunksize=1)
            # results = self.pool.imap(_subject_attachment_and_regularity, args, chunksize=int(len(args)/self.number_of_processes))
            logger.debug('time taken for deformations : ' + str(time.perf_counter() - start))

            # Sum and return.
            if with_grad:
                attachment = 0.0
                regularity = 0.0

                gradient = {}
                if not self.freeze_template:
                    gradient['momenta_t'] = np.zeros(self.fixed_effects['momenta_t'].shape)
                if not self.freeze_control_points:
                    gradient['control_points'] = np.zeros(self.fixed_effects['control_points'].shape)
                if not self.freeze_momenta:
                    gradient['momenta'] = np.zeros(self.fixed_effects['momenta'].shape)

                for result in results:
                    i, (attachment_i, regularity_i, gradient_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                    for key, value in gradient_i.items():
                        if key == 'momenta':
                            gradient[key][i] = value
                        else:
                            gradient[key] += value
                return attachment, regularity, gradient
            else:
                attachment = 0.0
                regularity = 0.0
                for result in results:
                    i, (attachment_i, regularity_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                return attachment, regularity

        else:
            device, device_id = get_best_device(gpu_mode=self.gpu_mode)
            hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta, \
            momenta_t, module_intensities, module_positions, module_variances = self._fixed_effects_to_torch_tensors(with_grad,device=device)
            sparse_matrix = self.construct_sparse_matrix(template_points['image_points'], module_positions, module_variances, module_intensities)
            return self._compute_attachment_and_regularity(dataset, hypertemplate_data, hypertemplate_points, control_points,
                                                           momenta, momenta_t, sparse_matrix, module_intensities, module_positions, module_variances, with_grad, device=device)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def _deform_and_compute_attachment_and_regularity(exponential, template_points, control_points, momenta, module_positions, module_variances, module_intensities, sparse_matrix,
                                                      template, template_data,
                                                      multi_object_attachment, deformable_objects,
                                                      objects_noise_variance, regu_var_m, regu_var_m_ortho,
                                                      device='cpu'):

        # Deform.
        exponential.set_initial_template_points(template_points)
        exponential.set_initial_control_points(control_points)
        exponential.set_initial_momenta(momenta)
        exponential.move_data_to_(device=device)
        exponential.update()

        # Compute attachment and regularity.
        deformed_points = exponential.get_template_points()
        deformed_data = template.get_deformed_data(deformed_points, template_data)
        deformed_data['image_intensities'] += sparse_matrix
        attachment = -multi_object_attachment.compute_weighted_distance(deformed_data, template, deformable_objects,
                                                                        objects_noise_variance)
        regularity = - exponential.get_norm_squared()

        # x = template_points['image_points'].cpu().detach()
        # final_cp = exponential.control_points_t[-1].cpu().detach()
        # regu = torch.zeros(x.shape[:-1], dtype=torch.float64)
        # for k in range(final_cp.shape[0]):
        #     m = momenta[k].clone().cpu().detach()
        #     if m.numpy().any():
        #         m /= torch.norm(m)
        #         e = torch.randn(m.size(), dtype=torch.float64)
        #         e -= e.dot(m) * m
        #         e /= torch.norm(e)
        #         if m.size()[0] == 2:
        #             y = x - final_cp[k]
        #             regu += torch.exp(-torch.mm(y.view(-1,2), m.view(2,1)) ** 2 / (2 * 1) - torch.mm(y.view(-1,2), e.view(2,1)) ** 2 / (2 * 10)).view(y.shape[:-1])
        #         else:
        #             e2 = torch.cross(m, e)
        #             y = x - final_cp[k]
        #             regu += torch.exp(
        #                 -torch.dot(y, m) ** 2 / (2 * 1) - torch.dot(y, e) ** 2 / (2 * 10) - torch.dot(y, e2) ** 2 / (2 * 10))
        #
        # dim = template_data['image_intensities'].shape
        # regu2 = torch.zeros(dim).double()
        # for k in range(module_positions.shape[0]):
        #     x_norm = torch.mul(x ** 2, 1 / module_variances[k] ** 2).sum(-1).view(-1, 1)
        #     y_norm = torch.mul(module_positions[k] ** 2, 1 / module_variances[k] ** 2).sum().view(-1, 1)
        #     points_divided = torch.mul(x, 1 / module_variances[k] ** 2)
        #     dist = (x_norm + y_norm - 2.0 * torch.mul(points_divided, module_positions[k]).sum(-1).view(-1,
        #                                                                                                      1)).reshape(
        #             dim)
        #     regu2 += torch.exp(-dist)*torch.abs(module_intensities[k].detach())
        #
        # regularity -= 200.*torch.sum(torch.mul(torch.tensor(regu, dtype=torch.float64),regu2))

        final_cp = exponential.control_points_t[-1]
        final_momenta = exponential.momenta_t[-1]
        for k in range(final_cp.shape[0]):
            if momenta[k].detach().numpy().any():
                m = final_momenta[k] / torch.norm(momenta[k])
                e = torch.randn(m.shape[0], dtype=torch.float64)
                e = e - torch.dot(e, m) * m
                e = e / torch.norm(e)
                for l in range(module_positions.shape[0]):
                    if m.size()[0] == 2:
                        y = module_positions[l] - final_cp[k]
                        regularity += - 10000*torch.sum(torch.exp(- torch.mm(y.view(-1,2), m.view(2,1)) ** 2 / (2 * regu_var_m) - torch.mm(y.view(-1,2), e.view(2,1)) ** 2 / (2 * regu_var_m_ortho)).view(y.shape[:-1]))
                    else:
                        e2 = torch.cross(m, e)
                        y = module_positions[l] - final_cp[k]
                        regularity += - torch.sum(torch.exp(
                            -torch.dot(y, m) ** 2 / (2 * regu_var_m) - torch.dot(y, e) ** 2 / (2 * regu_var_m_ortho) - torch.dot(y, e2) ** 2 / (2 * regu_var_m_ortho)).view(y.shape[:-1]))

        x_norm = (module_positions ** 2).sum(1).view(-1, 1)
        dist = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(module_positions, torch.transpose(module_positions, 0, 1))

        # dist = torch.zeros([16,16], dtype=torch.float64)
        # for k in range(16):
        #     dist[k,:] = torch.norm(module_positions - module_positions[k], dim=1)**2/400
        regularity -= -torch.sum(torch.exp(-dist))

        assert torch.device(
            device) == attachment.device == regularity.device, 'attachment and regularity tensors must be on the same device. ' \
                                                               'device=' + device + \
                                                               ', attachment.device=' + str(attachment.device) + \
                                                               ', regularity.device=' + str(regularity.device)

        return attachment, regularity

    @staticmethod
    def _compute_gradients(attachment, regularity,
                           freeze_template, momenta_t,
                           freeze_control_points, control_points,
                           freeze_momenta, momenta, freeze_sparse_matrix, module_intensities, module_positions, module_variances,
                           with_grad=False):
        if with_grad:
            total_for_subject = attachment + regularity
            total_for_subject.backward()

            gradient = {}
            if not freeze_template:
                assert momenta_t.grad is not None, 'Gradients have not been computed'
                gradient['momenta_t'] = momenta_t.grad.detach().cpu().numpy()
            if not freeze_control_points:
                assert control_points.grad is not None, 'Gradients have not been computed'
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if not freeze_momenta:
                assert momenta.grad is not None, 'Gradients have not been computed'
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()
            if not freeze_sparse_matrix:
                gradient['module_intensities'] = module_intensities.grad.detach().cpu().numpy()
                gradient['module_positions'] = module_positions.grad.detach().cpu().numpy()
                gradient['module_variances'] = module_variances.grad.detach().cpu().numpy()

            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

        return res

    def _compute_attachment_and_regularity(self, dataset, hypertemplate_data, hypertemplate_points, control_points, momenta, momenta_t, sparse_matrix, module_intensities, module_positions, module_variances,
                                           with_grad=False, device='cpu'):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output.
        Single-thread version.
        """

        # Initialize.
        targets = [target[0] for target in dataset.deformable_objects]
        attachment = 0.
        regularity = 0.

        self.exponential_t.set_initial_template_points(hypertemplate_points)
        self.exponential_t.set_initial_control_points(control_points)
        self.exponential_t.set_initial_momenta(momenta_t[0])
        self.exponential_t.move_data_to_(device=device)
        self.exponential_t.update()
        template_points = self.exponential_t.get_template_points()
        template_data = self.hypertemplate.get_deformed_data(template_points, hypertemplate_data)
        self.set_template_data({key: value.detach().cpu().numpy() for key, value in template_data.items()})

        regularity -= self.exponential_t.get_norm_squared()

        # loop for every deformable object
        # deform and update attachment and regularity
        for i, target in enumerate(targets):
            new_attachment, new_regularity = DeterministicAtlasWithModule._deform_and_compute_attachment_and_regularity(
                self.exponential, template_points, control_points, momenta[i], module_positions[i], module_variances[i], module_intensities[i], sparse_matrix[i],
                self.template, template_data, self.multi_object_attachment,
                target, self.objects_noise_variance, self.regu_var_m, self.regu_var_m_ortho,
                device=device)

            attachment += new_attachment
            regularity += new_regularity

        # Compute gradient.
        return self._compute_gradients(attachment, regularity,
                                       self.freeze_template, momenta_t,
                                       self.freeze_control_points, control_points,
                                       self.freeze_momenta, momenta, self.freeze_sparse_matrix, module_intensities, module_positions, module_variances,
                                       with_grad)

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def construct_sparse_matrix(self, points, module_centers, module_variances, module_intensities):
        dim = (self.number_of_subjects,) + self.fixed_effects['template_data']['image_intensities'].shape
        sparse_matrix = torch.zeros(dim).double()
        for i in range(dim[0]):
            for k in range(self.nb_modules):
                x_norm = torch.mul(points ** 2, 1/module_variances[i,k]**2).sum(-1).view(-1, 1)
                y_norm = torch.mul(module_centers[i,k] ** 2, 1/module_variances[i,k]**2).sum().view(-1, 1)
                points_divided = torch.mul(points, 1/module_variances[i,k]**2)
                dist = (x_norm + y_norm - 2.0 * torch.mul(points_divided, module_centers[i,k]).sum(-1).view(-1,1)).reshape(dim[1:])
                sparse_matrix[i] += torch.exp(-dist)*module_intensities[i,k]
                # sparse_matrix[i] += 70/81*(1-dist)**2*(dist<1).double()*module_intensities[i,k]

                # x_norm = (points ** 2)
                # y_norm = (module_centers[i,k] ** 2)
                # dist = (x_norm + y_norm - 2.0 * torch.mul(points, module_centers[i, k])).view(-1, 2)
                # rect = ((dist[:,0] < module_variances[i,k,0]) * (dist[:,1] < module_variances[i,k,1])).double()
                # f = torch.exp(-dist.sum(1)/10)
                # conv2 = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=1, bias=False)
                # conv2.weights = rect.reshape([1, 1, 100, 100])
                # sparse_matrix[i] += conv2(f.float().reshape([1,1,100,100]))[0,0,1:-1,1:-1].double()*module_intensities[i,k]

                #sparse_matrix[i] += module_intensities[i,k] * (1/(1+torch.exp(-2*2*(points[:,:,0] - module_centers[i,k,0] + module_variances[i,k,0]/2))) - 1/(1+torch.exp(-2*2*(points[:,:,0] - module_centers[i,k,0] - module_variances[i,k,0]/2)))) * (1/(1+ torch.exp(-2*2*(points[:,:,1] - module_centers[i,k,1] + module_variances[i,k,1]/2))) - 1/(1+torch.exp(-2*2*(points[:,:,1] - module_centers[i,k,1] - module_variances[i,k,1]/2))))

                # x_norm = torch.mul(points ** 2, 1/module_variances[i,k]**2).sum(2).view(-1, 1)
                # y_norm = torch.mul(module_centers[i,k] ** 2, 1/module_variances[i,k]**2).sum().view(-1, 1)
                # points_divided = torch.mul(points, 1/module_variances[i,k]**2)
                # dist = (x_norm + y_norm - 2.0 * torch.mul(points_divided, module_centers[i,k]).sum(2).view(-1,1)).reshape(dim[1:])
                # sparse_matrix[i] += module_intensities[i,k] * (1/(1+torch.exp(-2*100*(dist+0.1))) - 1/(1+torch.exp(-2*100*(dist - 1))))
        return sparse_matrix

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                  requires_grad=False)
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                    requires_grad=False)
                           for key, value in template_points.items()}

        hypertemplate_data = self.fixed_effects['hypertemplate_data']
        hypertemplate_data = {key: move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                  requires_grad=False)
                         for key, value in hypertemplate_data.items()}

        # Template points.
        hypertemplate_points = self.hypertemplate.get_points()
        hypertemplate_points = {key: move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                    requires_grad=False)
                           for key, value in hypertemplate_points.items()}

        momenta_t = self.fixed_effects['momenta_t']
        momenta_t = move_data(momenta_t, device=device, dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_template and with_grad))

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = move_data(control_points, device=device, dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_control_points and with_grad))
            # control_points = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type),
            #                           requires_grad=(not self.freeze_control_points and with_grad))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(not self.freeze_momenta and with_grad))

        module_intensities = self.fixed_effects['module_intensities']
        module_intensities = move_data(module_intensities, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(not self.freeze_sparse_matrix and with_grad))

        module_positions = self.fixed_effects['module_positions']
        module_positions = move_data(module_positions, device=device, dtype=self.tensor_scalar_type,
                                             requires_grad=(not self.freeze_sparse_matrix and with_grad))

        module_variances = self.fixed_effects['module_variances']
        module_variances = move_data(module_variances, device=device, dtype=self.tensor_scalar_type,
                                             requires_grad=(not self.freeze_sparse_matrix and with_grad))

        return hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta, momenta_t, module_intensities, module_positions, module_variances

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals]
            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):
        device, _ = get_best_device(self.gpu_mode)

        # Initialize.
        hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta, momenta_t, \
        module_intensities, module_positions, module_variances = self._fixed_effects_to_torch_tensors(False, device=device)

        sparse_matrix = self.construct_sparse_matrix(template_points['image_points'], module_positions, module_variances, module_intensities)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            # Writing the whole flow.
            names = []
            for k, object_name in enumerate(self.objects_name):
                name = self.name + '__flow__' + object_name + '__subject_' + subject_id
                names.append(name)
            #self.exponential.write_flow(names, self.objects_name_extension, self.template, template_data, output_dir)

            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            deformed_data['image_intensities'] += sparse_matrix[i]
            m = torch.max(deformed_data['image_intensities'])
            for k in range(module_positions.shape[1]):
                for j in range(module_positions[i,k].shape[0]):
                    module_positions[i,k,j] = min(99, module_positions[i,k,j])
                    module_positions[i, k, j] = max(0, module_positions[i, k, j])
                deformed_data['image_intensities'][tuple(module_positions[i,k].int())] = 2* m

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, dataset.deformable_objects[i][0]))

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)

            self.template.write(output_dir, names,
                                {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

            deformed_data['image_intensities'] = sparse_matrix[i]
            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + '_sparsematrix' + object_extension
                names.append(name)

            self.template.write(output_dir, names,
                                {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, output_dir):

        # Template.
        device, _ = get_best_device(self.gpu_mode)
        template_names = []
        hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta, momenta_t, module_intensities, module_positions, module_variances \
            = self._fixed_effects_to_torch_tensors(False, device=device)
        self.exponential_t.set_initial_template_points(hypertemplate_points)
        self.exponential_t.set_initial_control_points(control_points)
        self.exponential_t.set_initial_momenta(momenta_t[0])
        self.exponential_t.move_data_to_(device=device)
        self.exponential_t.update()
        template_points = self.exponential_t.get_template_points()
        template_data = self.hypertemplate.get_deformed_data(template_points, hypertemplate_data)
        self.set_template_data({key: value.detach().cpu().numpy() for key, value in template_data.items()})
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        write_2D_array(self.get_momenta_t()[0], output_dir, self.name + "__EstimatedParameters__Momenta_t.txt")
        write_3D_array(self.get_module_positions(), output_dir, self.name + "__EstimatedParameters__ModulePositions.txt")
        write_3D_array(self.get_module_intensities(), output_dir, self.name + "__EstimatedParameters__ModuleIntensities.txt")
        write_3D_array(self.get_module_variances(), output_dir, self.name + "__EstimatedParameters__ModuleVariances.txt")


