import glob
import logging
import math
import os
import os.path
import time
import warnings
from copy import deepcopy

import torch
from scipy.stats import norm
from torch.autograd import Variable

#import support.kernels as kernel_factory
#modif fleur
from ...support.kernels import factory

from ...core import default, GpuMode
from ...core.model_tools.deformations.piecewise_spatiotemporal_reference_frame import SpatiotemporalReferenceFrame
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta, initialize_modulation_matrix, \
    initialize_sources, \
    initialize_onset_ages, initialize_accelerations, initialize_covariance_momenta_inverse
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from ...support import utilities
from ...support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from ...support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from ...support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from ...support.probability_distributions.dirichlet_distribution import DirichletDistribution
from ...support.probability_distributions.normal_distribution import NormalDistribution
from ...support.probability_distributions.uniform_distribution import UniformDistribution


logger = logging.getLogger(__name__)


def compute_exponential_and_attachment(args):
    # Read inputs and restore the general settings.
    from .abstract_statistical_model import process_initial_data
    if process_initial_data is None:
        raise RuntimeError('process_initial_data is not set !')

    # start = time.perf_counter()

    # Read arguments.
    (template, multi_object_attachment, tensor_scalar_type, gpu_mode, exponential) = process_initial_data
    # (i, j, exponential, template_data, target, with_grad) = args
    (ijs, initial_template_points, initial_control_points, initial_momenta, template_data, targets, with_grad) = args

    ret_residuals = []
    ret_grad_template_points = []
    ret_grad_control_points = []
    ret_grad_momenta = []

    device, device_id = utilities.get_best_device(gpu_mode=gpu_mode)
    # device, device_id = ('cpu', -1)
    if device_id >= 0:
        torch.cuda.set_device(device_id)

    # create cuda streams
    # streams = []
    # for i in range(2):  # TODO: best value for number of streams
    #     streams.append(torch.cuda.Stream(device_id))

    # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
    # template = utilities.convert_deformable_object_to_torch(template, device=device)
    # template_data = {key: utilities.move_data(value, device=device) for key, value in template_data.items()}

    # torch.cuda.synchronize()    # wait for all data to be transferred to device

    assert len(ijs) == len(initial_template_points) == len(initial_control_points) == len(initial_momenta) == \
           len(targets), "should be the same size"

    for i in range(len(ijs)):
        # with torch.cuda.stream(streams[i % len(streams)]):
        # logger.info(">>>" + str(torch.cuda.current_stream()))

        exponential.set_initial_template_points(initial_template_points[i])
        exponential.set_initial_control_points(initial_control_points[i])
        exponential.set_initial_momenta(initial_momenta[i])
        exponential.move_data_to_(device)

        target = targets[i]
        # target = utilities.convert_deformable_object_to_torch(target, device=device)

        # Deform and compute the distance.
        if with_grad:
            exponential.initial_template_points = {key: value.requires_grad_() for key, value in
                                                   exponential.initial_template_points.items()}
            exponential.initial_control_points.requires_grad_()
            exponential.initial_momenta.requires_grad_()

        # start_update = time.perf_counter()
        exponential.move_data_to_(device)
        exponential.update()
        # logger.info('exponential.update(): ' + str(time.perf_counter() - start_update))

        deformed_points = exponential.get_template_points()
        # deformed_points = {'image_points': torch.rand(48, 65, 30, 3, device=device)}
        deformed_data = template.get_deformed_data(deformed_points, template_data)
        residual = multi_object_attachment.compute_distances(deformed_data, template, target)

        if with_grad:
            residual[0].backward()
            # ret_residuals[-1][0].backward()
            ret_grad_template_points.append(
                {key: value.grad.cpu() for key, value in exponential.initial_template_points.items()})
            ret_grad_control_points.append(exponential.initial_control_points.grad.cpu())
            ret_grad_momenta.append(exponential.initial_momenta.grad.cpu())

        ret_residuals.append(residual.detach().cpu())

    # wait for all streams to finish
    # for stream in streams:
    #     stream.synchronize()

    # torch.cuda.empty_cache()

    if with_grad:
        # compute gradients
        # residual[0].backward()
        # grad_template_points = {key: value.grad.cpu() for key, value in exponential.initial_template_points.items()}
        # grad_control_points = exponential.initial_control_points.grad.cpu()
        # grad_momenta = exponential.initial_momenta.grad.cpu()

        # logger.info('compute_exponential_and_attachment WITH grad: ' + str(time.perf_counter() - start))
        # return i, j, residual.cpu(), grad_template_points, grad_control_points, grad_momenta
        return ijs, ret_residuals, ret_grad_template_points, ret_grad_control_points, ret_grad_momenta
    else:
        # logger.info('compute_exponential_and_attachment WITHOUT grad: ' + str(time.perf_counter() - start))
        # return i, j, residual.cpu(), None, None, None
        return ijs, ret_residuals, None, None, None

    # start = time.perf_counter()
    #
    # device, device_id = utilities.get_best_device()
    # # device, device_id = ('cpu', -1)
    # if device_id >= 0:
    #     torch.cuda.set_device(device_id)
    #
    # # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
    # template = utilities.convert_deformable_object_to_torch(template, device=device)
    # exponential.move_data_to_(device)
    # template_data = {key: utilities.move_data(value, device=device) for key, value in template_data.items()}
    # target = utilities.convert_deformable_object_to_torch(target, device=device)
    #
    # # Deform and compute the distance.
    # if with_grad:
    #     exponential.initial_template_points = {key: value.requires_grad_() for key, value in exponential.initial_template_points.items()}
    #     exponential.initial_control_points.requires_grad_()
    #     exponential.initial_momenta.requires_grad_()
    #
    # # start_update = time.perf_counter()
    # exponential.update()
    # # logger.info('exponential.update(): ' + str(time.perf_counter() - start_update))
    #
    # deformed_points = exponential.get_template_points()
    # deformed_data = template.get_deformed_data(deformed_points, template_data)
    # residual = multi_object_attachment.compute_distances(deformed_data, template, target, device=device)
    #
    # if with_grad:
    #     # compute gradients
    #     residual[0].backward()
    #     grad_template_points = {key: value.grad.cpu() for key, value in exponential.initial_template_points.items()}
    #     grad_control_points = exponential.initial_control_points.grad.cpu()
    #     grad_momenta = exponential.initial_momenta.grad.cpu()
    #
    #     # logger.info('compute_exponential_and_attachment WITH grad: ' + str(time.perf_counter() - start))
    #     return i, j, residual.cpu(), grad_template_points, grad_control_points, grad_momenta
    # else:
    #     # logger.info('compute_exponential_and_attachment WITHOUT grad: ' + str(time.perf_counter() - start))
    #     return i, j, residual.cpu(), None, None, None


class ClusteredLongitudinalAtlas(AbstractStatisticalModel):
    """
    Longitudinal atlas object class.
    See "Learning distributions of shape trajectories from longitudinal datasets: a hierarchical model on a manifold
    of diffeomorphisms", B\^{o}ne et al. (2018), Computer Vision and Pattern Recognition conference.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,
                 gpu_mode=default.gpu_mode,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 concentration_of_time_points=default.concentration_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot,
                 use_rk2_for_flow=default.use_rk2_for_flow,
                 tR=default.t0,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 number_of_sources=default.number_of_sources,
                 initial_modulation_matrix=default.initial_modulation_matrix,
                 freeze_modulation_matrix=default.freeze_modulation_matrix,

                 freeze_rupture_time=default.freeze_reference_time,

                 initial_time_shift_variance=default.initial_time_shift_variance,

                 initial_acceleration_mean=default.initial_acceleration_mean,
                 initial_acceleration_variance=default.initial_acceleration_variance,
                 freeze_time_parameters_variance=default.freeze_acceleration_variance,

                 freeze_noise_variance=default.freeze_noise_variance,

                 nb_classes=1,
                 num_component = 'Erreur',
                 min_times = 0, max_times = 100,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='ClusteredLongitudinalAtlas', number_of_processes=number_of_processes, gpu_mode=gpu_mode)

        if gpu_mode not in [GpuMode.KERNEL]:
            logger.warning("LongitudinalAtlas model currently only accepts KERNEL gpu mode. Forcing KERNEL gpu mode.")
            self.gpu_mode = GpuMode.KERNEL

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode
        self.nb_classes = nb_classes
        self.num_component = num_component
        self.alpha = 0.5
        print('>> Parameter of the Dirichlet Distribution initialized at 0.5')

        #Declare the piecewise structure of the templates
        if num_component == 'Erreur':
            raise('Please, specify the piecewise properties of template in model.xml')
        #self.num_component[i][j] specify the index of the jth component of the ith class

        self.nb_tot_component = 0 #Total numbers of components to estimate
        self.nb_component = np.zeros(self.nb_classes) #array indicating the number of components of the template of each class
        self.nb_first_component = 0 #number of components begining at -infinity
        first_momenta = []
        for i in range(self.nb_classes):
            if self.num_component[i][0] not in first_momenta:
                self.nb_first_component += 1
                first_momenta.append(self.num_component[i][0])
            self.nb_component[i] = self.num_component[i].__len__()
            if max(self.num_component[i]) + 1 > self.nb_tot_component:
                self.nb_tot_component = max(self.num_component[i]) + 1
            self.nb_max_component = max(self.nb_component) #Maximum number of componenys for one class

        self.num_rupt = [None]*self.nb_classes #Indicate the index of the rupture times of each class
        self.num_rupt[0] = list(range(max(1,self.num_component[0].__len__() - 1)))
        self.nb_rupt_tot = max(1,self.num_component[0].__len__() - 1)
        self.is_first_rupture = [0]*self.nb_rupt_tot #Indicates if the rupture time indexed i is a rupture time to estimate (i.e. first rupture time of a piecewise geodesic)
        self.is_first_rupture[0] = 1
        for i in range(1,self.nb_classes):
            self.num_rupt[i] = []
            for l in range(max(1,self.num_component[i].__len__() - 1)):
                found = 0
                for j in range(i):
                    if self.num_component[i][l] == self.num_component[j][l]:
                        self.num_rupt[i].append(self.num_rupt[j][l])
                        found = 1
                        break
                if not found:
                    self.num_rupt[i].append(self.nb_rupt_tot)
                    self.nb_rupt_tot += 1
                    if l == 0: self.is_first_rupture.append(1)

        # Declare model structure
        self.fixed_effects['template_data'] = [None]*self.nb_classes
        self.fixed_effects['control_points'] = [None]*self.nb_classes
        self.fixed_effects['momenta'] = [None]*self.nb_classes
        for i in range(nb_classes):
            self.fixed_effects['momenta'][i] = [None]*int(self.nb_component[i])
        self.fixed_effects['modulation_matrix'] = [None]*self.nb_classes
        self.fixed_effects['rupture_time'] = [None]*self.nb_rupt_tot
        self.fixed_effects['time_parameters_variance'] = [None]*self.nb_classes
        # self.fixed_effects['acceleration_variance'] = None
        self.fixed_effects['noise_variance'] = None
        self.fixed_effects['classes_probability'] = 1/nb_classes*np.ones(nb_classes)

        self.is_frozen = {'template_data': freeze_template, 'control_points': freeze_control_points,
                          'momenta': freeze_momenta, 'modulation_matrix': freeze_modulation_matrix,
                          'rupture_time': freeze_rupture_time, 'time_parameters_variance': freeze_time_parameters_variance,
                          'noise_variance': freeze_noise_variance}

        self.priors['template_data'] = {}
        self.priors['control_points'] = MultiScalarNormalDistribution()
        self.priors['momenta'] = MultiScalarNormalDistribution()
        self.priors['modulation_matrix'] = MultiScalarNormalDistribution()
        self.priors['rupture_time'] = []
        for k in range(self.nb_rupt_tot):
            self.priors['rupture_time'].append(MultiScalarNormalDistribution())
        self.priors['time_parameters_variance'] = []
        for k in range(self.nb_classes):
            self.priors['time_parameters_variance'].append(InverseWishartDistribution())
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['classes_probability'] = DirichletDistribution()

        self.individual_random_effects['time_parameters'] = []
        for k in range(self.nb_classes):
            self.individual_random_effects['time_parameters'].append(NormalDistribution())
            self.individual_random_effects['time_parameters'][k].set_mean(np.zeros(int(self.nb_component[k] + 1)))
            self.individual_random_effects['time_parameters'][k].set_covariance(np.eye(int(self.nb_component[k] + 1)))
        self.individual_random_effects['sources'] = MultiScalarNormalDistribution()
        self.individual_random_effects['classes'] = UniformDistribution(max=self.nb_classes, proba=0)
        # self.individual_random_effects['onset_age'] = MultiScalarNormalDistribution()
        # self.individual_random_effects['acceleration'] = MultiScalarTruncatedNormalDistribution()

        # Deformation.
        self.spatiotemporal_reference_frame = []
        self.spatiotemporal_reference_frame_is_modified = []
        for k in range(self.nb_classes):
            self.spatiotemporal_reference_frame.append(SpatiotemporalReferenceFrame(
                dense_mode=dense_mode,
                kernel=kernel_factory.factory(deformation_kernel_type,
                                            gpu_mode=self.gpu_mode,
                                            kernel_width=deformation_kernel_width),
                shoot_kernel_type=shoot_kernel_type,
                concentration_of_time_points=concentration_of_time_points, number_of_time_points=number_of_time_points,
                tR=tR, use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow, template_tR=None,
                nb_components=self.nb_component[k], num_components=self.num_component[k]))
            self.spatiotemporal_reference_frame_is_modified.append(True)

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         objects_noise_variance, self.multi_object_attachment) = create_template_metadata(
            template_specifications, self.dimension, gpu_mode=self.gpu_mode)

        self.template = []
        for k in range(nb_classes):
            self.template.append(DeformableMultiObject(deepcopy(object_list)))
            # self.template[k].update()

        self.objects_noise_dimension = compute_noise_dimension(self.template[0], self.multi_object_attachment,
                                                                      self.dimension, self.objects_name)
        self.number_of_objects = len(self.template[0].object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type,
                                                         gpu_mode=self.gpu_mode,
                                                         kernel_width=smoothing_kernel_width)

        # Template data.
        for k in range(nb_classes):
            self.set_template_data(self.template[k].get_data(), k)
        self.__initialize_template_data_prior()

        # Control points.
        cp = initialize_control_points(
            initial_control_points, self.template[0], initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        for k in range(self.nb_classes):
            self.set_control_points(cp,k)
        self.number_of_control_points = len(self.fixed_effects['control_points'][0])
        self.__initialize_control_points_prior()

        # Momenta.
        for k in range(self.nb_classes):
            mom = []
            for l in range(int(self.nb_component[k])):
                mom.append(initialize_momenta(initial_momenta, self.number_of_control_points, self.dimension, random=False))
            self.set_momenta(mom,k)
        self.__initialize_momenta_prior()

        # Modulation matrix.
        self.number_of_sources = number_of_sources
        for k in range(nb_classes):
            self.fixed_effects['modulation_matrix'][k] = initialize_modulation_matrix(
                initial_modulation_matrix, self.number_of_control_points, self.number_of_sources, self.dimension)
        self.number_of_sources = self.get_modulation_matrix(0).shape[1]
        self.__initialize_modulation_matrix_prior()

        # Rupture time.
        time_range = max_times - min_times
        rupture = np.zeros(self.nb_rupt_tot)
        for i in range(self.nb_classes):
            k = 1
            for j in range(self.num_rupt[i].__len__()):
                if rupture[self.num_rupt[i][j]] == 0 or rupture[self.num_rupt[i][j]] > min_times + k*time_range/(self.nb_component[i]):
                    rupture[self.num_rupt[i][j]] = min_times + k*time_range/(self.nb_component[i])
                k += 1

        for k in range(self.nb_rupt_tot):
            self.set_rupture_time(rupture[k],k)
        self.__initialize_rupture_time_prior(2)

        self.__initialize_time_parameters_variance_prior()
        self.__initialize_classes_prior(0.5)

        # # Time-shift variance.
        # self.set_time_shift_variance(initial_time_shift_variance)
        # self.__initialize_time_shift_variance_prior()
        #
        # # Acceleration variance.
        # if initial_acceleration_variance is not None:
        #     self.set_acceleration_variance(initial_acceleration_variance)
        # else:
        #     acceleration_std = 1.5
        #     logger.info('>> The initial acceleration std fixed effect is ARBITRARILY set to %.1f.' % acceleration_std)
        #     self.set_acceleration_variance(acceleration_std ** 2)
        # self.__initialize_acceleration_variance_prior()

        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(objects_noise_variance)
        self.objects_noise_variance_prior_normalized_dof = [elt['noise_variance_prior_normalized_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Source random effect.
        assert self.number_of_sources is not None, \
            'Please specify the number of sources, or provide a modulation matrix file.'
        self.individual_random_effects['sources'].set_mean(np.zeros((self.number_of_sources,)))
        self.individual_random_effects['sources'].set_variance(1.0)

        self.individual_random_effects['classes'].set_max(self.nb_classes)

        # # Time-shift random effect.
        # assert self.individual_random_effects['onset_age'].mean is not None
        # assert self.individual_random_effects['onset_age'].variance_sqrt is not None
        #
        # # Acceleration random effect.
        # acceleration_mean = self.individual_random_effects['acceleration'].get_mean()
        # if initial_acceleration_mean is None:
        #     self.individual_random_effects['acceleration'].set_mean(np.ones((1,)))
        # elif isinstance(acceleration_mean, float):
        #     self.individual_random_effects['acceleration'].set_mean(np.ones((1,)) * acceleration_mean)

    def initialize_random_effects_realization(
            self, number_of_subjects,
            initial_sources=default.initial_sources,
            initial_onset_ages=default.initial_onset_ages,
            initial_accelerations=default.initial_accelerations,
            **kwargs):

        acc = initialize_accelerations(initial_accelerations, number_of_subjects)

        onset = np.zeros(number_of_subjects)
        classes = np.random.randint(0, self.nb_classes, number_of_subjects)
        for i in range(number_of_subjects):
            # onset[i] = np.mean(dataset.times[i])
            onset[i] = self.get_rupture_time(self.num_rupt[classes[i]][0])

        # Initialize the random effects realization.
        individual_RER = {
            'sources': initialize_sources(initial_sources, number_of_subjects, self.number_of_sources),
            'onset_ages': onset,
            'accelerations': np.log(acc.reshape([np.size(acc),1]))*np.ones([1,int(self.nb_max_component)]),
            'classes': classes
        }

        return individual_RER

    def initialize_noise_variance(self, dataset, individual_RER):
        # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
        for k, normalized_dof in enumerate(self.objects_noise_variance_prior_normalized_dof):
            dof = dataset.total_number_of_observations * normalized_dof * self.objects_noise_dimension[k]
            self.priors['noise_variance'].degrees_of_freedom.append(dof)

        if np.min(self.fixed_effects['noise_variance']) < 0.0:
            # Prior on the noise variance (inverse Wishart: scale scalars parameters).
            fixed_effects = self._fixed_effects_to_torch_tensors(False)
            momenta, control_points, modulation_matrix, rupture_times, template_data, template_points = self.reconstruct_class_parameters(
                fixed_effects)
            sources, onset_ages, accelerations, classes = self._individual_RER_to_torch_tensors(individual_RER, False)
            onset_subject, accel_subject = self._compute_time_shifts(onset_ages, accelerations, rupture_times,
                                                                     classes)
            absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations,
                                                                      rupture_times, classes)
            for k in range(self.nb_classes):
                self._update_spatiotemporal_reference_frame(template_points[k], control_points[k], momenta[k],
                                                            modulation_matrix[k], rupture_times[k], tmin, tmax, k)
            residuals, _, _ = self._compute_residuals(dataset, template_data, absolute_times, sources, classes)

            residuals_per_object = np.zeros((self.number_of_objects,))
            for i in range(len(residuals)):
                for j in range(len(residuals[i])):
                    residuals_per_object += residuals[i][j].detach().cpu().numpy()

            for k, scale_std in enumerate(self.objects_noise_variance_prior_scale_std):
                if scale_std is None:
                    self.priors['noise_variance'].scale_scalars.append(
                        0.01 * residuals_per_object[k] / self.priors['noise_variance'].degrees_of_freedom[k])
                else:
                    self.priors['noise_variance'].scale_scalars.append(scale_std ** 2)

            # New, more informed initial value for the noise variance.
            self.fixed_effects['noise_variance'] = np.array(self.priors['noise_variance'].scale_scalars)

        else:
            for k, object_noise_variance in enumerate(self.fixed_effects['noise_variance']):
                self.priors['noise_variance'].scale_scalars.append(object_noise_variance)

    def __initialize_template_data_prior(self):
        """
        Initialize the template data prior.
        """
        # If needed (i.e. template not frozen), initialize the associated prior.
        if not self.is_frozen['template_data']:
            template_data = self.get_template_data(0)

            for key, value in template_data.items():
                # Initialization.
                self.priors['template_data'][key] = MultiScalarNormalDistribution()

                # Set the template data prior mean as the initial template data.
                self.priors['template_data'][key].mean = value.copy()

                if key == 'landmark_points':
                    # Set the template data prior standard deviation to the deformation kernel width.
                    self.priors['template_data'][key].set_variance_sqrt(
                        self.spatiotemporal_reference_frame[0].get_kernel_width())
                elif key == 'image_intensities':
                    # Arbitrary value.
                    std = 0.5
                    logger.info('Template image intensities prior std parameter is ARBITRARILY set to %.3f.' % std)
                    self.priors['template_data'][key].set_variance_sqrt(std)

    def __initialize_control_points_prior(self):
        """
        Initialize the control points prior.
        """
        # If needed (i.e. control points not frozen), initialize the associated prior.
        if not self.is_frozen['control_points']:
            # Set the control points prior mean as the initial control points.
            self.priors['control_points'].set_mean(self.get_control_points(0).copy())
            # Set the control points prior standard deviation to the deformation kernel width.
            self.priors['control_points'].set_variance_sqrt(self.spatiotemporal_reference_frame[0].get_kernel_width())

    def __initialize_momenta_prior(self):
        """
        Initialize the momenta prior.
        """
        # If needed (i.e. momenta not frozen), initialize the associated prior.
        if not self.is_frozen['momenta']:
            # Set the momenta prior mean as the initial momenta.
            self.priors['momenta'].set_mean(self.get_momenta(0,0).copy())
            # Set the momenta prior variance as the norm of the initial rkhs matrix.
            assert self.spatiotemporal_reference_frame[0].get_kernel_width() is not None
            rkhs_matrix = initialize_covariance_momenta_inverse(
                self.fixed_effects['control_points'][0], self.spatiotemporal_reference_frame[0].exponential.kernel,
                self.dimension)
            self.priors['momenta'].set_variance(1. / np.linalg.norm(rkhs_matrix))  # Frobenius norm.
            logger.info('>> Momenta prior std set to %.3E.' % self.priors['momenta'].get_variance_sqrt())

    def __initialize_modulation_matrix_prior(self):
        """
        Initialize the modulation matrix prior.
        """
        # If needed (i.e. modulation matrix not frozen), initialize the associated prior.
        if not self.is_frozen['modulation_matrix']:
            # Set the modulation_matrix prior mean as the initial modulation_matrix.
            self.priors['modulation_matrix'].set_mean(self.get_modulation_matrix(0).copy())
            # Set the modulation_matrix prior standard deviation to the deformation kernel width.
            self.priors['modulation_matrix'].set_variance_sqrt(self.spatiotemporal_reference_frame[0].get_kernel_width())

    def __initialize_rupture_time_prior(self, initial_rupture_time_variance):
        """
        Initialize the reference time prior.
        """
        # If needed (i.e. reference time not frozen), initialize the associated prior.
        if not self.is_frozen['rupture_time']:
            for k in range(self.nb_rupt_tot):
                # Set the rupture_time prior mean as the initial reference_time.
                self.priors['rupture_time'][k].set_mean(np.zeros((1,)) + self.get_rupture_time(k).copy())
                # Check that the reference_time prior variance has been set.
                self.priors['rupture_time'][k].set_variance(initial_rupture_time_variance)

    # def __initialize_time_shift_variance_prior(self):
    #     """
    #     Initialize the time-shift variance prior.
    #     """
    #     # If needed (i.e. time-shift variance not frozen), initialize the associated prior.
    #     if not self.is_frozen['time_shift_variance']:
    #         # Set the time_shift_variance prior scale to the initial time_shift_variance fixed effect.
    #         self.priors['time_shift_variance'].scale_scalars.append(self.get_time_shift_variance())
    #         # Arbitrarily set the time_shift_variance prior dof to 1.
    #         logger.info('>> The time shift variance prior degrees of freedom parameter is ARBITRARILY set to 1.')
    #         self.priors['time_shift_variance'].degrees_of_freedom.append(1.0)
    #
    # def __initialize_acceleration_variance_prior(self):
    #     """
    #     Initialize the acceleration variance prior.
    #     """
    #     # If needed (i.e. acceleration variance not frozen), initialize the associated prior.
    #     if not self.is_frozen['acceleration_variance']:
    #         # Set the acceleration_variance prior scale to the initial acceleration_variance fixed effect.
    #         self.priors['acceleration_variance'].scale_scalars.append(self.get_acceleration_variance())
    #         # Arbitrarily set the acceleration_variance prior dof to 1.
    #         logger.info('>> The acceleration variance prior degrees of freedom parameter is ARBITRARILY set to 1.')
    #         self.priors['acceleration_variance'].degrees_of_freedom.append(1.0)

    def __initialize_time_parameters_variance_prior(self):
        for k in range(self.nb_classes):
            A = 0.5*np.eye(int(self.nb_component[k] + 1))
            A[-1,-1] = 5
            self.priors['time_parameters_variance'][k].set_scale_matrix(A)
            self.priors['time_parameters_variance'][k].degrees_of_freedom = 1
            logger.info('>> The time parameters variance prior degrees of freedom parameter is ARBITRARILY set to 1.')

    def __initialize_classes_prior(self, alpha):
        self.priors['classes_probability'].alpha = alpha

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self, c):
        return self.fixed_effects['template_data'][c]

    def set_template_data(self, td, c):
        self.fixed_effects['template_data'][c] = td
        self.template[c].set_data(td)
        self.spatiotemporal_reference_frame_is_modified[c] = True

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self,c):
        return self.fixed_effects['control_points'][c]

    def set_control_points(self, cp, c):
        self.fixed_effects['control_points'][c] = cp
        self.spatiotemporal_reference_frame_is_modified[c] = True

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self, c, component):
        return self.fixed_effects['momenta'][c][component]

    def set_momenta(self, mom, c):
        self.fixed_effects['momenta'][c] = mom
        self.spatiotemporal_reference_frame_is_modified[c] = True

    # Modulation matrix ------------------------------------------------------------------------------------------------
    def get_modulation_matrix(self, c):
        return self.fixed_effects['modulation_matrix'][c]

    def set_modulation_matrix(self, mm, c):
        self.fixed_effects['modulation_matrix'][c] = mm
        self.spatiotemporal_reference_frame_is_modified[c] = True

    # Reference time ---------------------------------------------------------------------------------------------------
    def get_rupture_time(self, index):
        return self.fixed_effects['rupture_time'][index]

    def set_rupture_time(self, rt, index):
        self.fixed_effects['rupture_time'][index] = np.float64(rt)
        if self.is_first_rupture[index]:
            for i in range(self.nb_classes):
                if self.num_rupt[i][0] == index:
                    A = np.zeros(int(self.nb_component[i] + 1))
                    A[-1] = rt
                    self.individual_random_effects['time_parameters'][i].set_mean(A)
                    self.spatiotemporal_reference_frame_is_modified[i] = True

    # # Time-shift variance ----------------------------------------------------------------------------------------------
    # def get_time_shift_variance(self):
    #     return self.fixed_effects['time_shift_variance']
    #
    # def set_time_shift_variance(self, tsv):
    #     self.fixed_effects['time_shift_variance'] = np.float64(tsv)
    #     self.individual_random_effects['onset_age'].set_variance(tsv)
    #
    # # Log-acceleration variance ----------------------------------------------------------------------------------------
    # def get_acceleration_variance(self):
    #     return self.fixed_effects['acceleration_variance']
    #
    # def set_acceleration_variance(self, lav):
    #     self.fixed_effects['acceleration_variance'] = np.float64(lav)
    #     self.individual_random_effects['acceleration'].set_variance(lav)

    # Time parameters variance------------------------------------------------------------------------------------------
    def get_time_parameters_variance(self, c):
        return self.individual_random_effects['time_parameters'][c].covariance

    def set_time_parameters_variance(self, tsv, c):
        self.fixed_effects['time_parameters_variance'][c] = tsv
        self.individual_random_effects['time_parameters'][c].set_covariance(tsv)

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    #Classes probability
    def set_classes_probability(self,w):
        self.individual_random_effects['classes'].set_probability(w)
        self.fixed_effects['classes_probability'] = w

    # Class 2 fixed effects --------------------------------------------------------------------------------------------
    def get_fixed_effects(self, mode='class2'):
        out = {}

        if mode == 'class2':
            new_component = []
            for k in range(self.nb_classes):
                for l in range(int(self.nb_component[k])):
                    index = self.num_component[k][l]
                    if index not in new_component:
                        new_component.append(index)
                        if l == 0:
                            for key, value in self.fixed_effects['template_data'][k].items():
                                out[key + str(index)] = value
                            out['control_points' + str(index)] = self.fixed_effects['control_points'][k]
                        elif l > 1:
                            out['rupture_time' + str(self.num_rupt[k][l - 1])] = np.array(
                                self.fixed_effects['rupture_time'][self.num_rupt[k][l - 1]])
                        out['momenta' + str(index)] = self.fixed_effects['momenta'][k][l]
                out['modulation_matrix' + str(k)] = self.fixed_effects['modulation_matrix'][k]

        elif mode == 'all':
            print('To Do')
        #     for key, value in self.fixed_effects['template_data'].items():
        #         out[key] = value
        #     out['control_points'] = self.fixed_effects['control_points']
        #     out['momenta'] = self.fixed_effects['momenta']
        #     out['modulation_matrix'] = self.fixed_effects['modulation_matrix']
        #     out['reference_time'] = self.fixed_effects['reference_time']
        #     out['time_shift_variance'] = self.fixed_effects['time_shift_variance']
        #     out['acceleration_variance'] = self.fixed_effects['acceleration_variance']
        #     out['noise_variance'] = self.fixed_effects['noise_variance']

        return out

    def set_fixed_effects(self, fixed_effects):
        for k in range(self.nb_classes):
            mom = []
            for l in range(int(self.nb_component[k])):
                index = self.num_component[k][l]
                mom.append(fixed_effects['momenta' + str(index)])

            index = self.num_component[k][0]
            if 'landmark_points' in self.fixed_effects['template_data'][0].keys():
                template_data = {key: fixed_effects['landmark_points' + str(index)] for key, value in
                                self.fixed_effects['template_data'][0].items()}
                self.set_template_data(template_data, k)
            elif 'image_intensities' in self.fixed_effects['template_data'][0].keys():
                template_data = {key: fixed_effects['image_intensities' + str(index)] for key, value in
                                self.fixed_effects['template_data'][0].items()}
                self.set_template_data(template_data, k)
            self.set_control_points(fixed_effects['control_points' + str(index)], k)
            self.set_momenta(mom, k)
            self.set_modulation_matrix(fixed_effects['modulation_matrix' + str(k)], k)
            for l in self.num_rupt[k][1:]:
                self.set_rupture_time(fixed_effects['rupture_time' + str(l)], l)

    # For brute optimization of the longitudinal registration model ----------------------------------------------------
    # def get_parameters_variability(self):
    #     """
    #     Only to be called in the case of brute optimization of a longitudinal registration model.
    #     """
    #     assert (self.is_frozen['template_data'] and self.is_frozen['control_points'] and self.is_frozen['momenta'] and
    #             self.is_frozen['modulation_matrix'] and self.is_frozen['reference_time'] and
    #             self.is_frozen['time_shift_variance'] and self.is_frozen['acceleration_variance'] and
    #             self.is_frozen['noise_variance']), \
    #         'Error: the get_parameters_variability should only be called when estimating a longitudinal ' \
    #         'registration model, with the grid search algorithm.'
    #     out = {
    #         'acceleration': 5.0 * np.sqrt(self.get_acceleration_variance()),
    #         'onset_age': 5.0 * np.sqrt(self.get_time_shift_variance()),
    #         'sources': 5.0 * np.ones((self.number_of_sources,))
    #     }
    #     return out
    #
    # def get_parameters_bounds(self):
    #     """
    #     Only to be called in the case of brute optimization of a longitudinal registration model.
    #     """
    #     assert (self.is_frozen['template_data'] and self.is_frozen['control_points'] and self.is_frozen['momenta'] and
    #             self.is_frozen['modulation_matrix'] and self.is_frozen['reference_time'] and
    #             self.is_frozen['time_shift_variance'] and self.is_frozen['acceleration_variance'] and
    #             self.is_frozen['noise_variance']), \
    #         'Error: the get_parameters_bounds should only be called when estimating a longitudinal ' \
    #         'registration model, with the grid search algorithm.'
    #     out = {
    #         'acceleration': (0.0, None),
    #         'onset_age': (None, None),
    #         'sources': (None, None)
    #     }
    #     return out

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def setup_multiprocess_pool(self, dataset):
        self._setup_multiprocess_pool(initargs=(
            self.template[0], self.multi_object_attachment, self.tensor_scalar_type, self.gpu_mode,
            self.spatiotemporal_reference_frame[0].exponential))

    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False,
                               modified_individual_RER='all'):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.
        Start by updating the class 1 fixed effects.

        :param dataset: LongitudinalDataset instance
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        torch.autograd.set_detect_anomaly(True)

        device, _ = utilities.get_best_device(self.gpu_mode)
        fixed_effects = self._fixed_effects_to_torch_tensors(with_grad, device=device)
        momenta, control_points, modulation_matrix, rupture_times, template_data, template_points = self.reconstruct_class_parameters(
            fixed_effects, device=device)

        sources, onset_ages, accelerations, classes = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete', device=device)
        onset_subject, accel_subject = self._compute_time_shifts(onset_ages, accelerations, rupture_times,
                                                                 classes)

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        # Compute residuals.
        absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations, rupture_times, classes)
        for k in range(self.nb_classes):
            self._update_spatiotemporal_reference_frame(template_points[k], control_points[k], momenta[k],
                                                        modulation_matrix[k], rupture_times[k], tmin, tmax, k,
                                                        modified_individual_RER=modified_individual_RER, device=device)

        residuals, checkpoints_tensors, grad_checkpoints_tensors = self._compute_residuals(
            dataset, template_data, absolute_times, sources, classes, with_grad=with_grad)

        # # Update the fixed effects only if the user asked for the complete log likelihood.
        # if mode == 'complete':
        #     sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
        #                                                                residuals=residuals)
        #     self.update_fixed_effects(dataset, sufficient_statistics)

        # Compute the attachment, with the updated noise variance parameter in the 'complete' mode.
        attachments, grad_checkpoints_tensors = self._compute_individual_attachments(residuals,
                                                                                     grad_checkpoints_tensors)
        attachment = torch.sum(attachments)

        # Compute the regularity terms according to the mode.
        regularity = utilities.move_data(np.array(0.0), dtype=self.tensor_scalar_type, device=device)
        if mode == 'complete':
            regularity = torch.sum(self._compute_random_effects_regularity(sources, onset_ages, accelerations, classes, rupture_times, device=device))
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, control_points, momenta,
                                                                 modulation_matrix)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            start = time.perf_counter()
            # Call backward.
            if self.number_of_processes == 1:
                total = attachment + regularity
                total.backward()
            else:
                torch.autograd.backward(
                    checkpoints_tensors + [regularity],
                    grad_checkpoints_tensors + [torch.ones(regularity.size(),
                                                           device=regularity.device, dtype=regularity.dtype)])

            logger.debug('time taken for backwards: ' + str(time.perf_counter() - start))

            # Construct the dictionary containing all gradients.
            gradient = {}
            for key, value in fixed_effects.items():
                if key[:-1] == 'landmark_points' and not self.is_frozen['template_data']:
                    for k in range(self.nb_classes):
                        if self.num_component[k][0] == int(key[-1]): break
                    gradient[key] = template_points[k][key[:-1]].grad
                elif key[:-1] == 'image_intensities':
                    for k in range(self.nb_classes):
                        if self.num_component[k][0] == int(key[-1]): break
                    gradient[key] = template_data[k][key[:-1]].grad
                elif key[:-1] != 'image_points' and not self.is_frozen[key[:-1]]:
                    gradient[key] = value.grad
                if self.use_sobolev_gradient and key[:-1] == 'landmark_points':
                    gradient[key] = self.sobolev_kernel.convolve(value.detach(), value.detach(), gradient[key].detach())

            if mode == 'complete':
                gradient['sources'] = sources.grad
                gradient['onset_age'] = onset_ages.grad
                gradient['acceleration'] = accelerations.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.detach().cpu().numpy() for key, value in gradient.items()}

            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient
            elif mode == 'model':
                return attachments.detach().cpu().numpy(), gradient

        else:
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()
            elif mode == 'model':
                return attachments.detach().cpu().numpy()

    def reconstruct_class_parameters(self, fixed_effects, device='cpu'): #returns an array giving the parameters defining each class
        momenta = []
        control_points = []
        modulation_matrix = []
        rupture_time = [None]*self.nb_classes
        template_data = []
        template_points = []
        for k in range(self.nb_classes):
            index = self.num_component[k][0]
            control_points.append(fixed_effects['control_points' + str(index)])
            template_data.append({})
            template_points.append({})
            # if 'landmark_points' in self.fixed_effects['template_data'][0].keys():
            #     for key,value in self.fixed_effects['template_data'][0].items():
            #         template_data[k][key] = fixed_effects[key + str(index)]
            #         template_points[k][key] = fixed_effects[key + str(index)]
            # elif 'image_intensities' in self.fixed_effects['template_data'][0].keys():
            for key,value in self.fixed_effects['template_data'][0].items():
                template_data[k][key] = fixed_effects[key + str(index)]
            for key,value in self.template[0].get_points().items():
                template_points[k][key] = fixed_effects[key + str(index)]

            mom = []
            for l in range(int(self.nb_component[k])):
                index = self.num_component[k][l]
                if l == 1 and index == self.num_component[k][0]:
                    mom.append(-fixed_effects['momenta' + str(index)]) #before the first rupture time
                else:
                    mom.append(fixed_effects['momenta' + str(index)])
            momenta.append(mom)
            modulation_matrix.append(fixed_effects['modulation_matrix' + str(k)])

            # rupture_time[k] = [utilities.move_data(np.array(self.get_rupture_time(self.num_rupt[k][0])),
            #                                      dtype=self.tensor_scalar_type, requires_grad=False, device=device)]
            rupture_time[k] = np.array([self.get_rupture_time(self.num_rupt[k][0])])

            for l in range(1,self.num_rupt[k].__len__()):
                rupture_time[k] = np.append(rupture_time[k],fixed_effects['rupture_time' + str(self.num_rupt[k][l])])

        return momenta, control_points, modulation_matrix, rupture_time, template_data, template_points

    def _compute_time_shifts(self, onset, accelerations, rupture_times, classes): #Compute the time shifts and accelerations of each subject on each component (fixed by continuity)
        accel_subject = []
        onset_subject = []
        for i in range(onset.__len__()):
            classe_i = int(classes[i])
            accel_subject_i = []
            for k in range(self.num_component[classe_i].__len__()):
                accel_subject_i.append(accelerations[i][k])
            accel_subject.append(accel_subject_i)
            onset_i = []
            for k in range(int(self.nb_component[classe_i])):
                if k < 2: #Same onset on the first two components
                    onset_i.append(onset[i])
                else: #Fixed by continuity
                    onset_i.append(onset_i[k-1] + (rupture_times[classe_i][k-1] - rupture_times[classe_i][k-2])/np.exp(accel_subject[i][k-2]))
            onset_subject.append(onset_i)
        return onset_subject, accel_subject

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None, model_terms=None):
        """
        Compute the model sufficient statistics.
        """
        sufficient_statistics = {}
        number_of_subjects = dataset.number_of_subjects

        sufficient_statistics['S1'] = np.zeros(self.nb_classes)
        sufficient_statistics['S2'] = np.zeros(1)
        for i in range(number_of_subjects):
            sufficient_statistics['S1'][individual_RER['classes'][i]] += 1
            sufficient_statistics['S2'][0] += dataset.times[i].__len__()

        fixed_effects = self._fixed_effects_to_torch_tensors(False)
        momenta, control_points, modulation_matrix, rupture_time, template_data, template_points = self.reconstruct_class_parameters(
            fixed_effects)
        sources, onset_ages, accelerations, classes = self._individual_RER_to_torch_tensors(
            individual_RER, False)
        onset_subject, accel_subject = self._compute_time_shifts(onset_ages, accelerations, rupture_time,
                                                                 classes)

        # Second statistical moment of the residuals (most costy part).
        if not self.is_frozen['noise_variance']:
            sufficient_statistics['S3'] = np.zeros((self.number_of_objects,))

            # Trick to save useless computations. Could be extended to work in the multi-object case as well ...
            if model_terms is not None and self.number_of_objects == 1:
                sufficient_statistics['S3'][0] += - 2 * np.sum(model_terms) * self.get_noise_variance()

            # Standard case.
            else:
                if residuals is None:
                    absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations,
                                                                              rupture_time, classes)
                    for k in range(self.nb_classes):
                        self._update_spatiotemporal_reference_frame(template_points[k], control_points[k],
                                                                    momenta[k],
                                                                    modulation_matrix[k], rupture_time[k],
                                                                    tmin, tmax, k)  # Problem if with_grad ?

                    residuals, _, _ = self._compute_residuals(dataset, template_data, absolute_times, sources, classes,
                                                              with_grad=False)

                for i in range(len(residuals)):
                    for j in range(len(residuals[i])):
                        for k in range(self.number_of_objects):
                            sufficient_statistics['S3'][k] += residuals[i][j][k].detach().cpu().numpy()

        rupture = self.fixed_effects['rupture_time']
        inv_sigma = []
        sigma = []
        for k in range(self.nb_classes):
            inv_sigma.append(self.individual_random_effects['time_parameters'][k].covariance_inverse)
            sigma.append(self.individual_random_effects['time_parameters'][k].covariance)

        sufficient_statistics['S4'] = []
        for k in range(self.nb_classes):
            sufficient_statistics['S4'].append(np.zeros([int(self.nb_component[k] + 1), int(self.nb_component[k] + 1)]))
        sufficient_statistics['S5'] = np.zeros(self.nb_rupt_tot)
        sufficient_statistics['S6'] = np.zeros(self.nb_rupt_tot)
        for i in range(number_of_subjects):
            cl = classes[i]
            num_rupt = self.num_rupt[classes[i]][0]

            time_parameters = []
            centered_time_parameters = []
            for k in range(accel_subject[i].__len__()):
                time_parameters.append(individual_RER['accelerations'][i][k])
                centered_time_parameters.append(individual_RER['accelerations'][i][k])
            centered_time_parameters.append(individual_RER['onset_ages'][i] - rupture[num_rupt])
            time_parameters.append(individual_RER['onset_ages'][i])

            sufficient_statistics['S4'][cl] += np.dot(np.array(centered_time_parameters).reshape([time_parameters.__len__(), 1]),
                                np.array(centered_time_parameters).reshape([1, time_parameters.__len__()]))
            sufficient_statistics['S6'][num_rupt] += inv_sigma[cl][-1, -1]
            for l in range(time_parameters.__len__()):
                sufficient_statistics['S5'][num_rupt] += inv_sigma[cl][-1, l] * time_parameters[l]

        sufficient_statistics['S4'] = np.array(sufficient_statistics['S4'])

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        number_of_subjects = dataset.number_of_subjects
        # Link update of the first rupture times and the time parameters covariance
        continuer = True
        inv_sigma = []
        sigma = []
        rupture = self.fixed_effects['rupture_time'].copy()

        for k in range(self.nb_classes):
            inv_sigma.append(self.individual_random_effects['time_parameters'][k].covariance_inverse)
            sigma.append(self.individual_random_effects['time_parameters'][k].covariance)

        while continuer:
            old_rupture = rupture.copy()
            old_sigma = sigma.copy()

            for k in range(self.nb_classes):
                prior_scale_matrix = self.priors['time_parameters_variance'][k].scale_matrix
                prior_dofs = self.priors['time_parameters_variance'][k].degrees_of_freedom
                sigma[k] = (sufficient_statistics['S4'][k] + prior_scale_matrix * prior_dofs) / (
                            prior_dofs + sufficient_statistics['S1'][k])
                inv_sigma[k] = np.linalg.inv(sigma[k])

            for k in range(self.nb_rupt_tot):
                reftime_prior_mean = self.priors['rupture_time'][k].mean
                reftime_prior_variance = self.priors['rupture_time'][k].variance_sqrt ** 2
                rupture[k] = (reftime_prior_variance * sufficient_statistics['S5'][k] + reftime_prior_mean) / (
                        sufficient_statistics['S6'][k] * reftime_prior_variance + 1)

            if np.linalg.norm(np.array(rupture) - np.array(old_rupture)) / np.linalg.norm(old_rupture) < 1e-3:
                for k in range(self.nb_classes):
                    if np.linalg.norm(np.array(sigma[k]) - np.array(old_sigma[k])) / np.linalg.norm(
                            old_sigma[k]) > 1e-3:
                        continuer = True
                        break
                    else:
                        continuer = False

        for k in range(self.nb_rupt_tot):
            if sufficient_statistics['S6'][k] != 0:  # we only maximize the first rupture times
                self.set_rupture_time(rupture[k], k)
        for k in range(self.nb_classes):
            self.set_time_parameters_variance(sigma[k], k)

        # Update of the residual noise variance ------------------------------------------------------------------------
        if not self.is_frozen['noise_variance']:
            prior_scale_scalars = self.priors['noise_variance'].scale_scalars
            prior_dofs = self.priors['noise_variance'].degrees_of_freedom
            noise_variance = np.zeros((self.number_of_objects,))
            for k in range(self.number_of_objects):
                noise_variance[k] = \
                    (sufficient_statistics['S3'][k] + prior_scale_scalars[k] * prior_dofs[k]) \
                    / float(sufficient_statistics['S2'][0] * self.objects_noise_dimension[k] + prior_dofs[k])
            self.set_noise_variance(noise_variance)

        w = (np.array(sufficient_statistics['S1']) + self.priors['classes_probability'].alpha) / (
                    number_of_subjects + self.priors['classes_probability'].alpha * self.nb_classes)
        self.set_classes_probability(w)

    def preoptimize(self, dataset, individual_RER):

        logger.info('-------------------------------')
        a1, r1 = self.compute_log_likelihood(dataset, None, individual_RER)
        ll1 = a1 + r1

        # Removes the mean of the accelerations. -----------------------------------------------------------------------
        expected_mean_acceleration = self.individual_random_effects['acceleration'].get_expected_mean()
        mean_acceleration = np.mean(individual_RER['acceleration'])
        individual_RER['acceleration'] *= expected_mean_acceleration / mean_acceleration
        self.set_momenta(self.get_momenta() * mean_acceleration / expected_mean_acceleration)

        # # Remove the mean of the sources. ------------------------------------------------------------------------------
        # mean_sources = torch.from_numpy(np.mean(individual_RER['sources'], axis=0)).type(self.tensor_scalar_type)
        # individual_RER['sources'] -= (1. - factor) + factor * mean_sources
        #
        # # Initialization.
        # (template_data, template_points, control_points,
        #  momenta, modulation_matrix) = self._fixed_effects_to_torch_tensors(False)
        #
        # projected_modulation_matrix = torch.zeros(modulation_matrix.size()).type(self.tensor_scalar_type)
        # norm_squared = self.spatiotemporal_reference_frame.exponential.scalar_product(control_points, momenta, momenta)
        # for s in range(self.number_of_sources):
        #     sp = self.spatiotemporal_reference_frame.exponential.scalar_product(
        #         control_points, momenta, modulation_matrix[:, s].view(control_points.size())) / norm_squared
        #     projected_modulation_matrix[:, s] = modulation_matrix[:, s] - sp * momenta.view(-1)
        #
        # # Move template
        # space_shift = torch.mm(projected_modulation_matrix, mean_sources.unsqueeze(1)).view(control_points.size())
        # self.spatiotemporal_reference_frame.exponential.set_use_rk2_for_shoot(True)
        # self.spatiotemporal_reference_frame.exponential.set_initial_template_points(template_points)
        # self.spatiotemporal_reference_frame.exponential.set_initial_control_points(control_points)
        # self.spatiotemporal_reference_frame.exponential.set_initial_momenta(space_shift * factor)
        # self.spatiotemporal_reference_frame.exponential.update()
        # deformed_control_points = self.spatiotemporal_reference_frame.exponential.control_points_t[-1]
        # self.set_control_points(deformed_control_points.detach().cpu().numpy())
        # deformed_points = self.spatiotemporal_reference_frame.exponential.get_template_points()
        # deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        # self.set_template_data({key: value.detach().cpu().numpy() for key, value in deformed_data.items()})
        #
        # # Parallel transport the momenta.
        # self.set_momenta(self.spatiotemporal_reference_frame.exponential.parallel_transport(
        #     momenta, is_orthogonal=True)[-1].detach().cpu().numpy())
        #
        # # Parallel transport of the modulation matrix.
        # for s in range(self.number_of_sources):
        #     projected_modulation_matrix[:, s] = self.spatiotemporal_reference_frame.exponential.parallel_transport(
        #         projected_modulation_matrix[:, s].view(control_points.size()))[-1].view(-1)
        #
        # # Finalization.
        # self.set_modulation_matrix(projected_modulation_matrix.detach().cpu().numpy())
        # self.spatiotemporal_reference_frame.exponential.set_use_rk2_for_shoot(False)

        # # Remove the standard deviation of the sources. ----------------------------------------------------------------
        # std_sources = np.std(individual_RER['sources'], axis=0)
        # individual_RER['sources'] *= (1. - factor) + factor / std_sources
        # self.set_modulation_matrix(self.get_modulation_matrix() * ((1. - factor) + factor * std_sources))

        logger.info('-------------------------------')
        a2, r2 = self.compute_log_likelihood(dataset, None, individual_RER)
        ll2 = a2 + r2
        logger.info('-------------------------------')
        logger.info('a2 - a1 = %.5f' % (a2 - a1))
        logger.info('r2 - r1 = %.5f' % (r2 - r1))
        logger.info('ll2 - ll1 = %.5f' % (ll2 - ll1))
        logger.info('-------------------------------')
        logger.info('-------------------------------')

    ####################################################################################################################
    ### Private key methods:
    ####################################################################################################################

    def _compute_attachment(self, residuals):
        """
        Fully torch.
        """
        return torch.sum(self._compute_individual_attachments(residuals))

    def _compute_individual_attachments(self, residuals, grad_checkpoints_tensors=None):
        """
        Fully torch.
        """
        number_of_subjects = len(residuals)
        device = residuals[0][0].device

        attachments = torch.zeros((number_of_subjects,), dtype=self.tensor_scalar_type.dtype, device=device)
        noise_variance = utilities.move_data(self.fixed_effects['noise_variance'], dtype=self.tensor_scalar_type, device=device)

        for i in range(number_of_subjects):
            attachment_i = 0.0
            for j in range(len(residuals[i])):
                attachment_i -= 0.5 * torch.sum(residuals[i][j] / noise_variance)
            attachments[i] = attachment_i

        if self.number_of_processes > 1:
            assert grad_checkpoints_tensors is not None
            grad_checkpoints_tensors = [- 0.5 * elt / noise_variance
                                        for elt in grad_checkpoints_tensors]

        return attachments, grad_checkpoints_tensors

    def _compute_random_effects_regularity(self, sources, onset_ages, accelerations, classes, device='cpu'):
        """
        Fully torch.
        """
        number_of_subjects = onset_ages.shape[0]
        regularity = 0.0

        # Sources random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['sources'].compute_log_likelihood_torch(
                sources[i], self.tensor_scalar_type, device=device)

        # Time parameters random effect.
            time_parameters = []
            for k in range(accelerations[i].__len__()):
                time_parameters.append(float(accelerations[i][k]))
            time_parameters.append(float(onset_ages[i]))
            regularity += self.individual_random_effects['time_parameters'][classes[i]].compute_log_likelihood(
                np.array(time_parameters))

            # Classes regularity
            regularity += np.log(self.individual_random_effects['classes'].proba[classes[i]])

        # # Noise random effect (if not frozen).
        # if not self.is_frozen['noise_variance']:
        #     for k in range(self.number_of_objects):
        #         regularity -= 0.5 * self.objects_noise_dimension[k] * total_number_of_observations * math.log(
        #             self.fixed_effects['noise_variance'][k])

        return regularity

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Reference time prior (if not frozen).
        if not self.is_frozen['rupture_time']:
            for k in range(self.fixed_effects['rupture_time'].__len__()):
                regularity += self.priors['rupture_time'][k].compute_log_likelihood(self.fixed_effects['rupture_time'][k])

        # Time-parameters variance prior (if not frozen).
        if not self.is_frozen['time_parameters_variance']:
            for k in range(self.nb_classes):
                regularity += self.priors['time_parameters_variance'][k].compute_log_likelihood(self.fixed_effects['time_parameters_variance'][k])

        #Classes probability prior
        regularity += self.priors['classes_probability'].compute_log_likelihood(self.fixed_effects['classes_probability'])

        # Noise variance prior (if not frozen).
        if not self.is_frozen['noise_variance']:
            regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, template_data, control_points, momenta, modulation_matrix):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        all_index = []
        for k in range(self.nb_classes):
            index = self.num_component[k][0]

            if index not in all_index:
                # Prior on template_data fixed effects (if not frozen).
                if not self.is_frozen['template_data']:
                    for key, value in template_data[k].items():
                        regularity += self.priors['template_data'][key].compute_log_likelihood_torch(
                            value, self.tensor_scalar_type)

                # Prior on control_points fixed effects (if not frozen).
                if not self.is_frozen['control_points']:
                    regularity += self.priors['control_points'].compute_log_likelihood_torch(
                        control_points[k], self.tensor_scalar_type)

                # Prior on momenta fixed effects (if not frozen).
                if not self.is_frozen['momenta']:
                    for l in range(1,momenta[k].__len__()):
                            regularity += self.priors['momenta'].compute_log_likelihood_torch(momenta[k][l],
                                                                                              self.tensor_scalar_type)

            # Prior on modulation_matrix fixed effects (if not frozen).
            if not self.is_frozen['modulation_matrix']:
                regularity += self.priors['modulation_matrix'].compute_log_likelihood_torch(
                    modulation_matrix[k], self.tensor_scalar_type)

            all_index.append(index)

        return regularity

    def clear_memory(self):
        """
        Called by the srw_mhwg_sampler if a ValueError is detected. Useful if the geodesic had been extended before
        a problematic parallel transport.
        """
        self.spatiotemporal_reference_frame_is_modified = [True]*self.nb_classes

    def _update_spatiotemporal_reference_frame(self, template_points, control_points, momenta, modulation_matrix, tR,
                                               tmin, tmax, c, modified_individual_RER='all', device='cpu'):
        """
        Tries to optimize the computations, by avoiding repetitions of shooting / flowing / parallel transporting.
        If modified_individual_RER is None or that self.spatiotemporal_reference_frame_is_modified is True,
        no particular optimization is carried.
        In the opposite case, the spatiotemporal reference frame will be more subtly updated.
        """

        # logger.info('self.spatiotemporal_reference_frame_is_modified', self.spatiotemporal_reference_frame_is_modified)
        # t1 = time.time()

        if self.spatiotemporal_reference_frame_is_modified[c]:
            self.spatiotemporal_reference_frame[c].set_template_points_tR(template_points)
            self.spatiotemporal_reference_frame[c].set_control_points_tR(control_points)
            self.spatiotemporal_reference_frame[c].set_momenta(momenta)
            self.spatiotemporal_reference_frame[c].set_modulation_matrix_tR(modulation_matrix)
            self.spatiotemporal_reference_frame[c].set_tR(tR)
            self.spatiotemporal_reference_frame[c].set_tmin(tmin)
            self.spatiotemporal_reference_frame[c].set_tmax(tmax)
            self.spatiotemporal_reference_frame[c].update()

        else:
            if modified_individual_RER in ['onset_ages', 'accelerations', 'all']:
                self.spatiotemporal_reference_frame[c].set_tmin(tmin, optimize=True)
                self.spatiotemporal_reference_frame[c].set_tmax(tmax, optimize=True)
                self.spatiotemporal_reference_frame[c].update()

            elif not modified_individual_RER in ['sources', 'classes']:
                raise RuntimeError('Unexpected modified_individual_RER: "' + str(modified_individual_RER) + '"')

        self.spatiotemporal_reference_frame_is_modified[c] = False

        # t2 = time.time()
        # logger.info('>> Total time           : %.3f seconds' % (t2 - t1))

    def _compute_residuals(self, dataset, template_data, absolute_times, sources, classes, with_grad=True):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        targets = dataset.deformable_objects
        residuals = []  # List of list of torch 1D tensors. Individuals, time-points, object.

        checkpoint_tensors = []
        grad_checkpoint_tensors = []

        # if self.number_of_processes > 1 and not with_grad:
        if self.number_of_processes > 1:
            # Set arguments.
            args = []

            # TODO: check block size
            block_size = int(sum(len(x) for x in absolute_times) / self.number_of_processes)
            #block_size = 1

            tmp_ij = []
            tmp_initial_template_points = []
            tmp_initial_control_points = []
            tmp_initial_momenta = []
            tmp_targets = []

            for i in range(len(targets)):
                residuals_i = []
                for j, (absolute_time, target) in enumerate(zip(absolute_times[i], targets[i])):
                    residuals_i.append(None)

                    initial_template_points, initial_control_points, initial_momenta = \
                        self.spatiotemporal_reference_frame[classes[i]].get_template_points_exponential_parameters(
                            absolute_time, sources[i])

                    if with_grad:
                        #checkpoint_tensors += \
                        #    list(initial_template_points.values()) + [initial_control_points, initial_momenta]
                        checkpoint_tensors += \
                            list(initial_template_points.values()) + [initial_control_points, initial_momenta]

                    tmp_ij.append((i, j))
                    tmp_initial_template_points.append({key: value.detach()
                                                        for key, value in initial_template_points.items()})
                    tmp_initial_control_points.append(initial_control_points.detach())
                    tmp_initial_momenta.append(initial_momenta.detach())
                    tmp_targets.append(target)

                    if len(tmp_ij) == block_size:
                        args.append((tmp_ij, tmp_initial_template_points, tmp_initial_control_points,
                                     tmp_initial_momenta, template_data[classes[i]], tmp_targets, with_grad))

                        tmp_ij = []
                        tmp_initial_template_points = []
                        tmp_initial_control_points = []
                        tmp_initial_momenta = []
                        tmp_targets = []

                residuals.append(residuals_i)

            if len(tmp_ij) != 0:
                assert len(tmp_ij) == len(tmp_initial_template_points) == len(tmp_initial_control_points) == \
                       len(tmp_initial_momenta) == len(tmp_targets)
                args.append((tmp_ij, tmp_initial_template_points, tmp_initial_control_points,
                             tmp_initial_momenta, template_data, tmp_targets, with_grad))

                current_block_size = 0
                tmp_ij = []
                tmp_initial_template_points = []
                tmp_initial_control_points = []
                tmp_initial_momenta = []
                tmp_targets = []

            # Perform parallel computations
            start = time.perf_counter()
            results = self.pool.map(compute_exponential_and_attachment, args, chunksize=1)
            logger.debug('time taken to compute residuals: ' + str(time.perf_counter() - start) + ' for ' + str(
                len(args)) + ' tasks with a block_size of ' + str(block_size))

            # Gather results.
            for result in results:
                ijs, ret_residuals, grad_template_points, grad_control_points, grad_momentas = result

                if with_grad:
                    for (i, j), residual, grad_template_point, grad_control_point, grad_momenta \
                            in zip(ijs, ret_residuals, grad_template_points, grad_control_points, grad_momentas):
                        residuals[i][j] = residual

                        grad_checkpoint_tensors += list(grad_template_point.values()) + [grad_control_point,
                                                                                         grad_momenta]

                else:
                    for (i, j), residual in zip(ijs, ret_residuals):
                        residuals[i][j] = residual

                # i, j, residual, grad_template_points, grad_control_points, grad_momenta = result
                # residuals[i][j] = residual
                # if with_grad:
                #     grad_checkpoint_tensors += list(grad_template_points.values()) + [grad_control_points, grad_momenta]
        else:
            # logger.info('Perform sequential computations.')
            device, device_id = utilities.get_best_device(self.gpu_mode)
            start = time.perf_counter()

            # self.template = utilities.convert_deformable_object_to_torch(self.template, device=device)
            # self.template_data = {key: utilities.move_data(value, device=device) for key, value in
            #                       template_data.items()}

            for i in range(len(targets)):
                residuals_i = []
                for j, (absolute_time, target) in enumerate(zip(absolute_times[i], targets[i])):
                    # target = utilities.convert_deformable_object_to_torch(target, device=device)
                    deformed_points = self.spatiotemporal_reference_frame[classes[i]].get_template_points(
                        absolute_time, sources[i], device=device)
                    deformed_data = self.template[classes[i]].get_deformed_data(deformed_points, template_data[classes[i]])
                    residual = self.multi_object_attachment.compute_distances(deformed_data, self.template[classes[i]], target)
                    residuals_i.append(residual.cpu())
                residuals.append(residuals_i)

            logger.debug('time taken to compute residuals: ' + str(time.perf_counter() - start))

        assert len(checkpoint_tensors) == len(grad_checkpoint_tensors)
        return residuals, checkpoint_tensors, grad_checkpoint_tensors

    def _compute_absolute_times(self, times, onset_ages, accelerations, rupture_times, classes):
        """
        Fully torch.
        """
        rupture_subject, accel_subject = self._compute_time_shifts(onset_ages, accelerations, rupture_times, classes)
        # if acceleration_std > 1.0 and np.max(accelerations.data.cpu().numpy()) - 1.0 > 10.0 * acceleration_std:
        #     raise ValueError('Absurd numerical value for the acceleration factor: %.2f. Exception raised.'
        #                      'For reference, the acceleration std is %.2f.'
        #                      % (np.max(accelerations.data.cpu().numpy()), acceleration_std))

        absolute_times = []
        for i in range(len(times)):
            absolute_times_i = []
            for j in range(len(times[i])):
                t_ij = torch.from_numpy(np.array(times[i][j])).type(self.tensor_scalar_type)
                if t_ij < rupture_subject[i][0][0]:
                    absolute_times_i.append(
                        rupture_times[classes[i]] - torch.exp(torch.tensor(accel_subject[i][0])) * (-t_ij + rupture_subject[i][0][0]))
                else:
                    for l in range(1, rupture_subject[i].__len__()):
                        if l == rupture_subject[i].__len__() - 1:
                            absolute_times_i.append(rupture_times[classes[i]] + torch.exp(torch.tensor(accel_subject[i][l])) * (
                                        t_ij - rupture_subject[i][l][0]))
                        elif t_ij > rupture_subject[i][l] and t_ij < rupture_subject[i][l + 1]:
                            absolute_times_i.append(rupture_times[classes[i]] + torch.exp(accel_subject[i][l]) * (
                                        t_ij - rupture_subject[i][l][0]))
                            break
            absolute_times.append(absolute_times_i)

        tmin = min([subject_times[0].detach().cpu().numpy() for subject_times in absolute_times])
        tmax = max([subject_times[-1].detach().cpu().numpy() for subject_times in absolute_times])

        return absolute_times, tmin, tmax

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the input fixed_effects into torch tensors.
        """
        # Template data.
        fixed_effects = self.get_fixed_effects()
        for key, value in fixed_effects.items():
            if key[:-1] in ['landmark_points','image_intensities']: frozen = self.is_frozen['template_data']
            else: frozen = self.is_frozen[key[:-1]]
            fixed_effects[key] = utilities.move_data(value,
                                                 dtype=self.tensor_scalar_type,
                                                 requires_grad=with_grad and not frozen,
                                                 device=device)
        new_component = []
        for k in range(self.nb_classes):
            index = self.num_component[k][0]
            if index not in new_component:
                new_component.append(index)
                for key, value in self.template[k].get_points().items():
                    fixed_effects[key + str(index)] = utilities.move_data(value,
                                                 dtype=self.tensor_scalar_type,
                                                 requires_grad=with_grad and not self.is_frozen['template_data'],
                                                 device=device)

        return fixed_effects

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad, device='cpu'):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Sources.
        sources = individual_RER['sources']
        sources = utilities.move_data(sources, dtype=self.tensor_scalar_type, requires_grad=with_grad, device=device)
        # Onset ages.
        onset_ages = individual_RER['onset_ages']
        onset_ages = utilities.move_data(onset_ages, dtype=self.tensor_scalar_type, requires_grad=with_grad, device=device)
        # Accelerations.
        accelerations = individual_RER['accelerations']
        accelerations = utilities.move_data(accelerations, dtype=self.tensor_scalar_type, requires_grad=with_grad, device=device)

        classes = individual_RER['classes']
        return sources, onset_ages, accelerations, classes

    ####################################################################################################################
    ### Error handling methods:
    ####################################################################################################################

    def adapt_to_error(self, error):
        if error[:64] == 'Absurd required renormalization factor during parallel transport':
            self._augment_discretization()
        else:
            raise RuntimeError('Unknown response to the error: "%s"' % error)

    def _augment_discretization(self):
        for k in range(self.nb_classes):
            current_concentration = self.spatiotemporal_reference_frame[k].get_concentration_of_time_points()
            momenta_factor = current_concentration / float(current_concentration + 1)
            logger.info('Incrementing the concentration of time-points from %d to %d, and multiplying the momenta '
                  'by a factor %.3f.' % (current_concentration, current_concentration + 1, momenta_factor))
            self.spatiotemporal_reference_frame[k].set_concentration_of_time_points(current_concentration + 1)
            self.set_momenta(momenta_factor * self.get_momenta(k),k)

    ####################################################################################################################
    ### Printing and writing methods:
    ####################################################################################################################

    def print(self, individual_RER):
        logger.info('>> Model parameters:')

        # Noise variance.
        msg = '\t\t noise_std        ='
        noise_variance = self.get_noise_variance()
        for k, object_name in enumerate(self.objects_name):
            msg += '\t%.4f\t[ %s ]\t ; ' % (math.sqrt(noise_variance[k]), object_name)
        logger.info(msg[:-4])

        # Reference time, time-shift std, acceleration std.
        for k in range(self.nb_rupt_tot):
            logger.info('\t\t rupture_time of the class k   =\t%.3f' % self.get_rupture_time(k))
        # logger.info('\t\t time_shift_std   =\t%.3f' % math.sqrt(self.get_time_shift_variance()))
        # logger.info('\t\t acceleration_std =\t%.3f' % math.sqrt(self.get_acceleration_variance()))
        logger.info('\t\t classes   =\t '+ str(individual_RER['classes']))

        # Empirical distributions of the individual parameters.
        logger.info('>> Random effect empirical distributions:')
        logger.info('\t\t onset_ages       =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(individual_RER['onset_ages']), np.std(individual_RER['onset_ages'])))
        logger.info('\t\t accelerations    =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(individual_RER['accelerations']), np.std(individual_RER['accelerations'])))
        logger.info('\t\t sources          =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(individual_RER['sources']), np.std(individual_RER['sources'])))

        # Spatiotemporal reference frame length.
        for k in range(self.nb_classes):
            logger.info('>> Spatiotemporal reference frame length of the class k: %.2f.' %
                (self.spatiotemporal_reference_frame[k].get_tmax() - self.spatiotemporal_reference_frame[k].get_tmin()))

    def write(self, dataset, population_RER, individual_RER, output_dir, update_fixed_effects=False,
              write_residuals=True):
        self._clean_output_directory(output_dir)

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=(update_fixed_effects or write_residuals))

        # Optionally update the fixed effects.
        if update_fixed_effects:
            logger.info('Warning: not automatically updating the fixed effect.')
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals=residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Write residuals.
        if write_residuals:
            residuals_list = [[[residuals_i_j_k.detach().cpu().numpy() for residuals_i_j_k in residuals_i_j]
                               for residuals_i_j in residuals_i] for residuals_i in residuals]
            write_3D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(individual_RER, output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):

        # Initialize ---------------------------------------------------------------------------------------------------
        fixed_effects = self._fixed_effects_to_torch_tensors(False)
        momenta, control_points, modulation_matrix, rupture_time, template_data, template_points = self.reconstruct_class_parameters(
            fixed_effects)
        sources, onset_ages, accelerations, classes = self._individual_RER_to_torch_tensors(
            individual_RER, False)
        targets = dataset.deformable_objects
        absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations, rupture_time, classes)

        # Deform -------------------------------------------------------------------------------------------------------
        for k in range(self.nb_classes):
            self._update_spatiotemporal_reference_frame(template_points[k], control_points[k], momenta[k],
                                                        modulation_matrix[k], rupture_time[k], tmin, tmax, k)

            # Write --------------------------------------------------------------------------------------------------------
            # self.spatiotemporal_reference_frame[k].write(self.name, self.objects_name, self.objects_name_extension,
            #                                       self.template[k], template_data[k], output_dir)

        # Write reconstructions and compute residuals ------------------------------------------------------------------
        residuals = []  # List of list of torch 1D tensors. Individuals, time-points, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            residuals_i = []
            for j, (time, absolute_time) in enumerate(zip(dataset.times[i], absolute_times[i])):
                deformed_points = self.spatiotemporal_reference_frame[classes[i]].get_template_points(absolute_time, sources[i])
                deformed_data = self.template[classes[i]].get_deformed_data(deformed_points, template_data[classes[i]])

                if compute_residuals:
                    residuals_i.append(
                        self.multi_object_attachment.compute_distances(deformed_data, self.template[classes[i]], targets[i][j]))

                names = []
                for k, (object_name, object_extension) \
                        in enumerate(zip(self.objects_name, self.objects_name_extension)):
                    # name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id  + '__class_' + \
                    #        str(classes[i]) + ('__age_%.2f' % time) + '__tp_' + str(j) + object_extension
                    name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id  + '__class_' + \
                           str(classes[i]) + '__tp_' + str(j) + object_extension
                    names.append(name)
                self.template[classes[i]].write(output_dir, names,
                                    {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

            residuals.append(residuals_i)

        for k in range(self.nb_classes):
            names = []
            for j, time in enumerate(np.linspace(tmin, tmax, 20)):
                deformed_points = self.spatiotemporal_reference_frame[k].get_template_points(
                    torch.tensor(time).type(self.tensor_scalar_type),
                    torch.zeros(sources[0].__len__()).type(self.tensor_scalar_type))
                deformed_data = self.template[k].get_deformed_data(deformed_points, template_data[k])
                name = [self.name + '__Reconstruction__' + object_name + '__Template_classe_' + str(k) \
                            + '__tp_' + str(j) + object_extension]
                self.template[k].write(output_dir, name,
                                           {key: value.detach().cpu().numpy() for key, value in
                                            deformed_data.items()})

        return residuals

    def _write_model_parameters(self, individual_RER, output_dir):
        # Fixed effects ------------------------------------------------------------------------------------------------
        # Template.
        for c in range(self.nb_classes):
            template_names = []
            for k in range(len(self.objects_name)):
                aux = self.name + '__EstimatedParameters__Template_' + self.objects_name[k] + '__class_' + str(c) + '__tp_' \
                  + str(self.spatiotemporal_reference_frame[c].geodesic.exponential[0].number_of_time_points - 1) \
                  + ('__age_%.2f' % self.get_rupture_time(self.num_rupt[c][0])) + self.objects_name_extension[k]
            template_names.append(aux)
            self.template[c].write(output_dir, template_names)

            # Other class 1 fixed effects ----------------------------------------------------------------------------------
            write_2D_array(self.get_control_points(c), output_dir,
                           self.name + "__EstimatedParameters__ControlPoints" + "_classe" + str(c) + ".txt")
            for k in range(self.fixed_effects['momenta'][c].__len__()):
                write_3D_array(self.get_momenta(c, k), output_dir,
                               self.name + "__EstimatedParameters__Momenta" + str(k) + "_classe" + str(c) + ".txt")
            write_2D_array(self.get_modulation_matrix(c), output_dir,
                           self.name + "__EstimatedParameters__ModulationMatrix" + "_classe" + str(c) + ".txt")

        # Class 2 fixed effects ----------------------------------------------------------------------------------------
        for index in range(self.nb_rupt_tot):
            write_2D_array(np.zeros((1,)) + self.get_rupture_time(index), output_dir,
                        self.name + "__EstimatedParameters__RuptureTime__number_" + str(index) + ".txt")
        for l in range(self.nb_classes):
            write_3D_array(self.get_time_parameters_variance(l), output_dir,
                            self.name + "__EstimatedParameters__TimeParametersStd" + "_classe" + str(l) + ".txt")
        write_2D_array(np.sqrt(self.get_noise_variance()), output_dir,
                       self.name + "__EstimatedParameters__NoiseStd.txt")

        # Random effects realizations ----------------------------------------------------------------------------------
        # Sources.
        write_2D_array(individual_RER['sources'], output_dir, self.name + "__EstimatedParameters__Sources.txt")
        # Onset age.
        write_2D_array(individual_RER['onset_ages'], output_dir, self.name + "__EstimatedParameters__OnsetAges.txt")
        # Log-acceleration.
        write_2D_array(individual_RER['accelerations'], output_dir,
                       self.name + "__EstimatedParameters__Accelerations.txt")
        # Classes
        write_2D_array(individual_RER['classes'], output_dir,
                       self.name + "__EstimatedParameters__Classes.txt")

    def _clean_output_directory(self, output_dir):
        files_to_delete = glob.glob(output_dir + '/*')
        for file in files_to_delete:
            if not os.path.isdir(file) and (len(file) > 1 and not file[-2:] == '.p'):
                os.remove(file)
