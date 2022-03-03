import math

import torch

#import support.kernels as kernel_factory
#modif fleur
from ...support.kernels import factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_momenta, initialize_covariance_momenta_inverse, \
    initialize_control_points
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from ...support import utilities
from ...support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from ...support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from ...support.probability_distributions.normal_distribution import NormalDistribution
from ...support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from ...support.probability_distributions.laplace_distribution import LaplaceDistribution
from ...support.probability_distributions.alamain import AlamainDistribution
from ...support.probability_distributions.alamain_gradient import AlamainGradientDistribution
from ...support.probability_distributions.alamain_gradient_dependant import AlamainGradientDependantDistribution
from ...support.probability_distributions.alamain_gradient_allin import AlamainGradientAllInDistribution
from ...support.probability_distributions.multi_scalar_truncated_normal_distribution import MultiScalarTruncatedNormalDistribution
from ...core.model_tools.gaussian_smoothing import GaussianSmoothing
"""
from core import default
from core.model_tools.deformations.exponential import Exponential
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.models.model_functions import initialize_momenta, initialize_covariance_momenta_inverse, \
    initialize_control_points
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from support import utilities
from support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from support.probability_distributions.normal_distribution import NormalDistribution
from support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from support.probability_distributions.laplace_distribution import LaplaceDistribution
from support.probability_distributions.alamain import AlamainDistribution
from support.probability_distributions.alamain_gradient import AlamainGradientDistribution
from support.probability_distributions.alamain_gradient_dependant import AlamainGradientDependantDistribution
from support.probability_distributions.alamain_gradient_allin import AlamainGradientAllInDistribution
from support.probability_distributions.multi_scalar_truncated_normal_distribution import MultiScalarTruncatedNormalDistribution
from core.model_tools.gaussian_smoothing import GaussianSmoothing
"""

import logging
logger = logging.getLogger(__name__)


class BayesianAtlasSparse(AbstractStatisticalModel):
    """
    Bayesian atlas object class.
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

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 space_between_modules=20,
                 alpha_sparse=1000,

                 gpu_mode=default.gpu_mode,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='BayesianAtlas', gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode
        self.number_of_processes = number_of_processes
        self.space_between_modules = space_between_modules

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['covariance_momenta_inverse'] = None
        self.fixed_effects['noise_variance'] = None
        self.fixed_effects['momenta_t'] = None
        self.fixed_effects['sparse_matrix'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points

        self.priors['covariance_momenta'] = InverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()
        #self.priors['module_intensities'] = MultiScalarNormalDistribution()

        self.individual_random_effects['momenta'] = NormalDistribution()
        #self.individual_random_effects['module_variances'] = MultiScalarNormalDistribution()
        #self.individual_random_effects['module_directions'] = MultiScalarNormalDistribution()



        # Deformation.
        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=factory(deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        self.exponential_t = Exponential(
            dense_mode=dense_mode,
            kernel=factory(deformation_kernel_type, gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         objects_noise_variance, self.multi_object_attachment) = create_template_metadata(
            template_specifications, self.dimension, gpu_mode=gpu_mode)

        self.template = DeformableMultiObject(object_list)
        self.hypertemplate = DeformableMultiObject(object_list)
        # self.template.update()

        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                               self.dimension, self.objects_name)
        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = factory(deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()
        self.fixed_effects['hypertemplate_data'] = self.hypertemplate.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Covariance momenta.
        self.fixed_effects['covariance_momenta_inverse'] = initialize_covariance_momenta_inverse(
            self.fixed_effects['control_points'], self.exponential.kernel, self.dimension)
        self.priors['covariance_momenta'].scale_matrix = np.linalg.inv(self.fixed_effects['covariance_momenta_inverse'])*1000

        self.fixed_effects['momenta_t'] = initialize_momenta(
            default.initial_momenta, self.number_of_control_points, self.dimension, 1)

        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(objects_noise_variance)
        self.objects_noise_variance_prior_normalized_dof = [elt['noise_variance_prior_normalized_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Momenta random effect.
        self.individual_random_effects['momenta'].mean = np.zeros((self.number_of_control_points * self.dimension,))
        self.individual_random_effects['momenta'].set_covariance_inverse(
            self.fixed_effects['covariance_momenta_inverse']/1000)

        initial_cp = initialize_control_points(None, self.template, self.space_between_modules, None,self.dimension, False)
        self.number_of_modules = initial_cp.shape[0]

        #self.individual_random_effects['module_intensities'].mean = np.mean(np.array(self.template.get_data()['image_intensities']))*np.ones((2,))
        #self.individual_random_effects['module_intensities'].mean = np.array(0.)
        #self.individual_random_effects['module_intensities'].set_variance(50)
        self.gaussian_smoothing_var = 4

        image = self.template.get_data()['image_intensities']
        grad_x = image[2:, :] - image[:-2, :]
        grad_y = image[:, 2:] - image[:, :-2]
        grad_norm = np.multiply(grad_x, grad_x)[:,1:-1] + np.multiply(grad_y, grad_y)[1:-1,]

        smoothing = GaussianSmoothing(1, 1, self.gaussian_smoothing_var,image.ndim)
        shape = image.shape
        output = np.zeros(shape)
        if image.ndim == 2:
            regu = smoothing(
                torch.tensor(grad_norm.reshape(1, 1, shape[0] - 2, shape[1] - 2), dtype=torch.float32))[0, 0, :]
        else:
            regu = smoothing(
                torch.tensor(grad_norm.reshape(1, 1, shape[0] - 2, shape[1] - 2, shape[2]), dtype=torch.float32))[0, 0, :]
        begin_x = int((shape[0] - regu.shape[0]) / 2)
        begin_y = int((shape[1] - regu.shape[1]) / 2)
        output[begin_x:begin_x + regu.shape[0], begin_y:begin_y + regu.shape[1]] = regu

        for i in range(self.number_of_modules):
            self.individual_random_effects['module_' + str(i)] = AlamainGradientAllInDistribution()
            self.individual_random_effects['module_' + str(i)].box = (self.template.bounding_box)
            self.individual_random_effects['module_' + str(i)].alpha = alpha_sparse
            self.individual_random_effects['module_' + str(i)].points = self.template.get_points()['image_points']
            self.individual_random_effects['module_' + str(i)].set_image_grad(output)
            self.individual_random_effects['module_' + str(i)].mean_var = np.array(0.)
            self.individual_random_effects['module_' + str(i)].variance_var = np.array(1.)
            self.individual_random_effects['module_' + str(i)].mean_dir = np.array(0.)
            self.individual_random_effects['module_' + str(i)].variance_dir = np.array(100.)
            self.individual_random_effects['module_' + str(i)].mean_int = np.array(0.)
            self.individual_random_effects['module_' + str(i)].scale_int = np.array(10.)
            self.individual_random_effects['module_' + str(i)].dimension = self.dimension



    def initialize_random_effects_realization(
            self, number_of_subjects,
            initial_momenta=default.initial_momenta,
            covariance_momenta_prior_normalized_dof=default.covariance_momenta_prior_normalized_dof,
            **kwargs):
        initial_cp = initialize_control_points(None, self.template, self.space_between_modules, None,self.dimension, False)
        self.number_of_modules = initial_cp.shape[0]

        # Initialize the random effects realization.
        individual_RER = {
            'momenta': initialize_momenta(initial_momenta, self.number_of_control_points, self.dimension,
                                          number_of_subjects)
        }

        for i in range(self.number_of_modules):
            individual_RER['module_' + str(i)] = np.zeros([number_of_subjects,self.dimension*(self.dimension+1)  + 1])
            individual_RER['module_' + str(i)][:,:self.dimension] = np.array([initial_cp[i]]*number_of_subjects)
            individual_RER['module_' + str(i)][:,self.dimension+1:2*self.dimension+1] =  5*np.ones([number_of_subjects,self.dimension])
            individual_RER['module_' + str(i)][:,2*self.dimension+1:] = np.ones([number_of_subjects,(self.dimension-1)*self.dimension])
            self.individual_random_effects['module_' + str(i)].number_of_subjects = number_of_subjects


        # Initialize the corresponding priors.
        self.priors['covariance_momenta'].degrees_of_freedom = \
            number_of_subjects * covariance_momenta_prior_normalized_dof

        return individual_RER

    def initialize_noise_variance(self, dataset, individual_RER):
        # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
        for k, normalized_dof in enumerate(self.objects_noise_variance_prior_normalized_dof):
            dof = dataset.number_of_subjects * normalized_dof * self.objects_noise_dimension[k]
            self.priors['noise_variance'].degrees_of_freedom.append(dof)

        # Prior on the noise variance (inverse Wishart: scale scalars parameters).
        hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta_t = self._fixed_effects_to_torch_tensors(False)
        momenta, module_positions, module_intensities, module_variances, module_directions = self._individual_RER_to_torch_tensors(individual_RER, False)

        sparse_matrix = self.construct_sparse_matrix(template_points['image_points'], module_positions, module_variances, module_intensities, module_directions)
        residuals_per_object = sum(self._compute_residuals(
            dataset, template_data, template_points, control_points, momenta, sparse_matrix))
        for k, scale_std in enumerate(self.objects_noise_variance_prior_scale_std):
            if scale_std is None:
                self.priors['noise_variance'].scale_scalars.append(
                    0.01 * residuals_per_object[k].detach().cpu().numpy()
                    / self.priors['noise_variance'].degrees_of_freedom[k])
            else:
                self.priors['noise_variance'].scale_scalars.append(scale_std ** 2)

        # New, more informed initial value for the noise variance.
        self.fixed_effects['noise_variance'] = np.array(self.priors['noise_variance'].scale_scalars)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    def get_momenta_t(self):
        return self.fixed_effects['momenta_t']

    def set_momenta_t(self, mom):
        self.fixed_effects['momenta_t'] = mom

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp
        self.number_of_control_points = len(cp)

    # Covariance momenta inverse ---------------------------------------------------------------------------------------
    def get_covariance_momenta_inverse(self):
        return self.fixed_effects['covariance_momenta_inverse']

    def set_covariance_momenta_inverse(self, cmi):
        self.fixed_effects['covariance_momenta_inverse'] = cmi
        self.individual_random_effects['momenta'].set_covariance_inverse(cmi)

    def set_covariance_momenta(self, cm):
        self.set_covariance_momenta_inverse(np.linalg.inv(cm))

    # def set_intensity_classes(self, c):
    #     self.fixed_effects['intensity_classes'] = c
    #     self.individual_random_effects['module_intensities'].num_iter = -1
    #     self.individual_random_effects['module_intensities'].classes = c

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

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
    def get_fixed_effects(self, mode='class2'):
        out = {}

        if mode == 'class2':
            if not self.freeze_template:
                out['momenta_t'] = self.fixed_effects['momenta_t']
            if not self.freeze_control_points:
                out['control_points'] = self.fixed_effects['control_points']

        elif mode == 'all':
            out['momenta_t'] = self.fixed_effects['momenta_t']
            out['control_points'] = self.fixed_effects['control_points']
            out['covariance_momenta_inverse'] = self.fixed_effects['covariance_momenta_inverse']
            out['noise_variance'] = self.fixed_effects['noise_variance']

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            device, _ = utilities.get_best_device(self.gpu_mode)
            hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta_t \
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

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False,
                               modified_individual_RER='all'):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.
        Start by updating the class 1 fixed effects.

        :param dataset: LongitudinalDataset instance
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta_t = self._fixed_effects_to_torch_tensors(with_grad)
        momenta, module_positions, module_intensities, module_variances, module_directions = self._individual_RER_to_torch_tensors(
            individual_RER, False)
        sparse_matrix = self.construct_sparse_matrix(template_points['image_points'], module_positions,
                                                     module_variances, module_intensities, module_directions)

        device, _ = utilities.get_best_device(self.gpu_mode)
        self.exponential_t.set_initial_template_points(hypertemplate_points)
        self.exponential_t.set_initial_control_points(control_points)
        self.exponential_t.set_initial_momenta(momenta_t[0])
        self.exponential_t.move_data_to_(device=device)
        self.exponential_t.update()
        template_points = self.exponential_t.get_template_points()
        template_data = self.hypertemplate.get_deformed_data(template_points, hypertemplate_data)

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        residuals, image_grad = self._compute_residuals_and_image_grad(dataset, template_data, template_points, control_points, momenta, sparse_matrix)

        # Update the fixed effects only if the user asked for the complete log likelihood.
        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals=residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Compute the attachment, with the updated noise variance parameter in the 'complete' mode.
        attachments = self._compute_individual_attachments(residuals)
        attachment = torch.sum(attachments)

        # Compute the regularity terms according to the mode.
        regularity = torch.from_numpy(np.array(0.0)).type(self.tensor_scalar_type)
        if mode == 'complete':
            regularity = self._compute_random_effects_regularity(momenta, individual_RER['module_positions'], individual_RER['module_variances'], individual_RER['module_intensities'])
            regularity += self._compute_class1_priors_regularity()
            regularity += torch.norm(image_grad,2)
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(momenta_t, control_points)
            regularity += torch.norm(image_grad,2)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            if not self.freeze_template:
                gradient['momenta_t'] = momenta_t.grad.detach().cpu().numpy()
            if not self.freeze_control_points:
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if mode == 'complete':
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()
                gradient['sparse_matrix'] = sparse_matrix.grad.detach().cpu().numpy()

            # Return as appropriate.
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient
            elif mode == 'model':
                return attachments.detach().cpu().numpy(), gradient

        else:
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()
            elif mode == 'model':
                return attachments.detach().cpu().numpy()

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None, model_terms=None):
        """
        Compute the model sufficient statistics.
        """

        sufficient_statistics = {}

        # Empirical momenta covariance ---------------------------------------------------------------------------------
        momenta = individual_RER['momenta']
        sufficient_statistics['S1'] = np.zeros((momenta[0].size, momenta[0].size))
        for i in range(dataset.number_of_subjects):
            sufficient_statistics['S1'] += np.dot(momenta[i].reshape(-1, 1), momenta[i].reshape(-1, 1).transpose())

        #sufficient_statistics['S4'] = np.abs(individual_RER['module_intensities'] - self.individual_random_effects['module_intensities'].mean[1]) < np.abs(individual_RER['module_intensities'] - self.individual_random_effects['module_intensities'].mean[0])

        # sufficient_statistics['S3'] = np.zeros(2)
        # for i in range(dataset.number_of_subjects):
        #     for k in range(self.number_of_modules):
        #         if sufficient_statistics['S4'][i,k] == 0:
        #             sufficient_statistics['S3'][0] += individual_RER['module_intensities'][i,k]
        #         else:
        #             sufficient_statistics['S3'][1] += individual_RER['module_intensities'][i, k]

        # Empirical residuals variances, for each object ---------------------------------------------------------------
        sufficient_statistics['S2'] = np.zeros((self.number_of_objects,))

        hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta_t = self._fixed_effects_to_torch_tensors(
            False)
        momenta, module_positions, module_intensities, module_variances, module_directions = self._individual_RER_to_torch_tensors(
            individual_RER, False)
        sparse_matrix = self.construct_sparse_matrix(template_points['image_points'],
                                                     module_positions, module_variances, module_intensities, module_directions)


        #residuals, sufficient_statistics['S5'], momenta_t = self._compute_residuals_and_cp(dataset, template_data, template_points,
        #                                                                        control_points, momenta, sparse_matrix)

        residuals, sufficient_statistics['S5'] = self._compute_residuals_and_image_grad(dataset, template_data, template_points, control_points, momenta, sparse_matrix)
        residuals = [torch.sum(residuals_i) for residuals_i in residuals]

        #for i in range(dataset.number_of_subjects):
        #    self.individual_random_effects['module_positions_subj' + str(i)].module_directions = module_directions[i]
        #    self.individual_random_effects['module_positions_subj' + str(i)].module_intensities = module_intensities[i]
        #    self.individual_random_effects['module_positions_subj' + str(i)].module_variances = module_variances[i]

        # Trick to save useless computations. Could be extended to work in the multi-object case as well ...
        # if model_terms is not None and self.number_of_objects == 1:
        #     sufficient_statistics['S2'][0] += - 2 * np.sum(model_terms) * self.get_noise_variance()
        #     return sufficient_statistics

        # Standard case.
        for i in range(dataset.number_of_subjects):
            sufficient_statistics['S2'] += residuals[i].detach().cpu().numpy()

        # Return
        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        # Covariance of the momenta update.
        prior_scale_matrix = self.priors['covariance_momenta'].scale_matrix
        prior_dof = self.priors['covariance_momenta'].degrees_of_freedom
        covariance_momenta = (sufficient_statistics['S1'] + prior_dof * np.transpose(prior_scale_matrix)) \
                             / (dataset.number_of_subjects + prior_dof)
        self.set_covariance_momenta(covariance_momenta)

        # Variance of the residual noise update.
        noise_variance = np.zeros((self.number_of_objects,))
        prior_scale_scalars = self.priors['noise_variance'].scale_scalars
        prior_dofs = self.priors['noise_variance'].degrees_of_freedom
        for k in range(self.number_of_objects):
            noise_variance[k] = (sufficient_statistics['S2'][k] + prior_scale_scalars[k] * prior_dofs[k]) \
                                / float(dataset.number_of_subjects * self.objects_noise_dimension[k] + prior_dofs[k])
        self.set_noise_variance(noise_variance)

        # sigma_prior = self.priors['module_intensities'].get_variance_sqrt()**2
        # sigma_ind = self.individual_random_effects['module_intensities'].variance_sqrt**2
        # prior_mean = self.priors['module_intensities'].get_mean()
        # num = np.array([sum(sufficient_statistics['S4'].ravel() == 0),sum(sufficient_statistics['S4'].ravel() == 1)])
        # mean_intensities = (sigma_prior * sufficient_statistics['S3'] + sigma_ind * prior_mean)/(num*sigma_prior + sigma_ind)
        # self.individual_random_effects['module_intensities'].set_mean(mean_intensities)
        # self.set_intensity_classes(sufficient_statistics['S4'])

        for i in range(self.number_of_modules):
            self.individual_random_effects['module_' + str(i)].set_image_grad(sufficient_statistics['S5'])
                #self.individual_random_effects['module_positions_subj' + str(i)]

    def initialize_template_attributes(self, template_specifications):
        """
        Sets the Template, TemplateObjectsName, TemplateObjectsNameExtension, TemplateObjectsNorm,
        TemplateObjectsNormKernelType and TemplateObjectsNormKernelWidth attributes.
        """

        t_list, t_name, t_name_extension, t_noise_variance, t_multi_object_attachment = \
            create_template_metadata(template_specifications, gpu_mode=self.gpu_mode)

        self.template.object_list = t_list
        self.objects_name = t_name
        self.objects_name_extension = t_name_extension
        self.multi_object_attachment = t_multi_object_attachment

        self.template.update(self.dimension)
        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                               self.dimension)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment(self, residuals):
        """
        Fully torch.
        """
        return torch.sum(self._compute_individual_attachments(residuals))

    def _compute_individual_attachments(self, residuals):
        """
        Fully torch.
        """
        number_of_subjects = len(residuals)
        attachments = torch.zeros((number_of_subjects,)).type(self.tensor_scalar_type)
        for i in range(number_of_subjects):
            attachments[i] = - 0.5 * torch.sum(residuals[i] / utilities.move_data(
                self.fixed_effects['noise_variance'], dtype=self.tensor_scalar_type, device=residuals[i].device))
        return attachments

    def _compute_random_effects_regularity(self, momenta, control_points, module_positions):
        """
        Fully torch.
        """
        number_of_subjects = momenta.shape[0]
        regularity = 0.0

        # Momenta random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['momenta'].compute_log_likelihood_torch(
                momenta[i], self.tensor_scalar_type)

        # Noise random effect.
        for k in range(self.number_of_objects):
            regularity -= 0.5 * self.objects_noise_dimension[k] * number_of_subjects \
                          * math.log(self.fixed_effects['noise_variance'][k])

        return regularity

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Covariance momenta prior.
        regularity += self.priors['covariance_momenta'].compute_log_likelihood(
            self.fixed_effects['covariance_momenta_inverse'])

        # Noise variance prior.
        regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, template_data, control_points):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Prior on template_data fixed effects (if not frozen). None implemented yet TODO.
        if not self.freeze_template:
            regularity += 0.0

        # Prior on control_points fixed effects (if not frozen). None implemented yet TODO.
        if not self.freeze_control_points:
            regularity += 0.0

        return regularity

    def _compute_residuals(self, dataset, template_data, template_points, control_points, momenta, sparse_matrix):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        device, _ = utilities.get_best_device(self.exponential.kernel.gpu_mode)

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]

        # Deform -------------------------------------------------------------------------------------------------------
        residuals = []

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        for i, target in enumerate(targets):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.move_data_to_(device=device)
            self.exponential.update()
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            deformed_data['image_intensities'] += sparse_matrix[i]
            residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

        return residuals

    def _compute_residuals_and_image_grad(self, dataset, template_data, template_points, control_points, momenta, sparse_matrix):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        device, _ = utilities.get_best_device(self.exponential.kernel.gpu_mode)

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]

        # Deform -------------------------------------------------------------------------------------------------------
        residuals = []

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)
        image_grad = []

        for i, target in enumerate(targets):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.move_data_to_(device=device)
            self.exponential.update()
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            deformed_data['image_intensities'] += sparse_matrix[i]
            residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

            image = deformed_data['image_intensities']
            grad_x = image[2:, :] - image[:-2, :]
            grad_y = image[:, 2:] - image[:, :-2]
            grad_norm = np.multiply(grad_x, grad_x)[:, 1:-1] + np.multiply(grad_y, grad_y)[1:-1, ]

            smoothing = GaussianSmoothing(1, 1, self.gaussian_smoothing_var, image.numpy().ndim)
            shape = image.shape
            output = np.zeros(shape)
            if image.numpy().ndim == 2:
                regu = smoothing(
                    torch.tensor(grad_norm.reshape(1, 1, shape[0] - 2, shape[1] - 2), dtype=torch.float32))[0, 0, :]
            else:
                regu = smoothing(
                    torch.tensor(grad_norm.reshape(1, 1, shape[0] - 2, shape[1] - 2, shape[2]), dtype=torch.float32))[0,
                       0, :]

            begin_x = int((shape[0] - regu.shape[0]) / 2)
            begin_y = int((shape[1] - regu.shape[1]) / 2)
            output[begin_x:begin_x + regu.shape[0], begin_y:begin_y + regu.shape[1]] = regu
            image_grad.append(output)

        return residuals, np.array(image_grad)

    def _compute_residuals_and_cp(self, dataset, template_data, template_points, control_points, momenta, sparse_matrix):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        device, _ = utilities.get_best_device(self.exponential.kernel.gpu_mode)

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]

        # Deform -------------------------------------------------------------------------------------------------------
        residuals = []
        cp = np.zeros(momenta.shape)
        momenta_t = np.zeros(momenta.shape)

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        for i, target in enumerate(targets):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.move_data_to_(device=device)
            self.exponential.update()
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            deformed_data['image_intensities'] += sparse_matrix[i]
            residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))
            cp[i] = np.array(self.exponential.control_points_t[-1])
            momenta_t[i] = np.array(self.exponential.momenta_t[-1])

        return residuals, cp, momenta_t

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the input fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: torch.from_numpy(value).type(self.tensor_scalar_type).requires_grad_(
            not self.freeze_template and with_grad) for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: torch.from_numpy(value).type(self.tensor_scalar_type).requires_grad_(
            not self.freeze_template and with_grad) for key, value in template_points.items()}

        # Template data.
        hypertemplate_data = self.fixed_effects['hypertemplate_data']
        hypertemplate_data = {key: torch.from_numpy(value).type(self.tensor_scalar_type) for key, value in hypertemplate_data.items()}

        # Template points.
        hypertemplate_points = self.hypertemplate.get_points()
        hypertemplate_points = {key: torch.from_numpy(value).type(self.tensor_scalar_type) for key, value in hypertemplate_points.items()}

        momenta_t = self.get_momenta_t()
        momenta_t = utilities.move_data(momenta_t, device=device, dtype=self.tensor_scalar_type,
                                        requires_grad=(not self.freeze_template and with_grad))

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = torch.from_numpy(control_points).type(self.tensor_scalar_type).requires_grad_(
                not self.freeze_control_points and with_grad)

        return hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta_t

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Momenta.
        momenta = individual_RER['momenta']
        momenta = torch.from_numpy(momenta).type(self.tensor_scalar_type).requires_grad_(with_grad)

        module_positions = np.zeros([momenta.shape[0], self.number_of_modules, self.dimension])
        for i in range(self.number_of_modules):
            module_positions[:,i,:] = individual_RER['module_' + str(i)][:,:self.dimension]
        module_positions = torch.from_numpy(np.array(module_positions)).type(self.tensor_scalar_type).requires_grad_(with_grad)

        module_intensities = np.zeros([momenta.shape[0], self.number_of_modules])
        for i in range(self.number_of_modules):
            module_intensities[:,i] = individual_RER['module_' + str(i)][:,self.dimension]
        module_intensities = torch.from_numpy(np.array(module_intensities)).type(self.tensor_scalar_type).requires_grad_(with_grad)

        module_variances = np.zeros([momenta.shape[0], self.number_of_modules, self.dimension])
        for i in range(self.number_of_modules):
            module_variances[:,i,:] = individual_RER['module_' + str(i)][:,self.dimension+1:2*self.dimension+1]
        module_variances = torch.from_numpy(np.array(module_variances)).type(self.tensor_scalar_type).requires_grad_(with_grad)

        module_directions = np.ones([momenta.shape[0],self.number_of_modules,self.dimension-1, self.dimension])
        for i in range(self.number_of_modules):
            module_directions[:,i,:] = individual_RER['module_' + str(i)][:,2*self.dimension+1:].reshape([momenta.shape[0],self.dimension-1, self.dimension])
        module_directions = torch.from_numpy(np.array(module_directions)).type(self.tensor_scalar_type).requires_grad_(with_grad)

        return momenta, module_positions, module_intensities, module_variances, module_directions

    def construct_sparse_matrix(self, points, module_centers, module_variances, module_intensities, module_direction):
        dim = (module_intensities.shape[0],) + self.fixed_effects['template_data']['image_intensities'].shape
        sparse_matrix = torch.zeros(dim).double()
        for i in range(dim[0]):
            for k in range(self.number_of_modules):
                if not module_direction[i,k].detach().numpy().any():
                    module_direction[i,k,0] = 1
                dir = module_direction[i,k,0]/torch.norm(module_direction[i,k,0])
                if dir.shape[0] == 2:
                    e = torch.rand(dir.shape[0], dtype=torch.float64)
                    e -= torch.dot(e,dir)*dir
                    e /= torch.norm(e)
                else:
                    e = module_direction[i,k,1]/torch.norm(module_direction[i,k,1])
                    e -= torch.dot(e, dir) * dir
                    e /= torch.norm(e)
                    e2 = torch.cross(dir,e)

                y = points - module_centers[i,k]
                if dir.shape[0] == 2:
                    dist = (torch.mm(y.view(-1,2), dir.view(2,1))**2/module_variances[i,k,0]**2 + torch.mm(y.view(-1,2), e.view(2,1))**2/module_variances[i,k,1]**2).reshape(dim[1:])
                else:
                    dist = (torch.mm(y.view(-1,3), dir.view(3,1))**2/module_variances[i,k,0]**2 + torch.mm(y.view(-1,3), e.view(3,1))**2/module_variances[i,k,1]**2 + torch.mm(y.view(-1,3), e2.view(3,1))**2/module_variances[i,k,2]**2).reshape(dim[1:])

                # x_norm = torch.mul(points ** 2, 1/module_variances[i,k]**2).sum(-1)
                # y_norm = torch.mul(module_centers[i,k] ** 2, 1/module_variances[i,k]**2).sum()
                # points_divided = torch.mul(points, 1/module_variances[i,k]**2)
                # dist = (x_norm + y_norm - 2.0 * torch.mul(points_divided, module_centers[i,k]).sum(-1)).reshape(dim[1:])
                sparse_matrix[i] += torch.exp(-dist)*module_intensities[i,k]
        return sparse_matrix

    ####################################################################################################################
    ### Printing and writing methods:
    ####################################################################################################################

    def print(self, individual_RER):
        pass

    def write(self, dataset, population_RER, individual_RER, output_dir, update_fixed_effects=True,
              write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=(update_fixed_effects or write_residuals))

        # Optionally update the fixed effects.
        if update_fixed_effects:
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals=residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.detach().cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals]
            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(individual_RER, output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):
        device, _ = utilities.get_best_device(self.exponential.kernel.gpu_mode)

        # Initialize.
        hypertemplate_data, hypertemplate_points, template_data, template_points, control_points, momenta_t = self._fixed_effects_to_torch_tensors(False)
        momenta, module_positions, module_intensities, module_variances, module_directions = self._individual_RER_to_torch_tensors(
            individual_RER, False)
        sparse_matrix = self.construct_sparse_matrix(template_points['image_points'],
                                                     module_positions, module_variances, module_intensities, module_directions)
        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.move_data_to_(device=device)
            self.exponential.update()

            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            deformed_data['image_intensities'] += sparse_matrix[i]
            # m = torch.max(deformed_data['image_intensities'])
            # for k in range(self.number_of_modules):
            #     deformed_data['image_intensities'][tuple(module_positions[i,k].int())] = 2* m

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

            #max = np.max(np.array(deformed_data['image_intensities']))
            deformed_data['image_intensities'] = sparse_matrix[i]
            # for k in range(self.number_of_modules):
            #     add_coord = True
            #     for l in range(module_positions[i, k][0].shape[0]):
            #         if np.array(module_positions[i, k][0][l]).astype(int) >= sparse_matrix.shape[l]:
            #             add_coord = False
            #     if np.min(np.array(module_positions[i, k][0]).astype(int)) < 0:
            #         add_coord = False
            #     if add_coord:
            #         coord = np.array(module_positions[i, k][0]).astype(int)
            #         deformed_data['image_intensities'][coord] = max
            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + '_sparsematrix' + object_extension
                names.append(name)
            self.template.write(output_dir, names,
                                {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, individual_RER, output_dir):
        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(individual_RER['momenta'], output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        write_3D_array(self.get_momenta_t(), output_dir, self.name + "__EstimatedParameters__Momenta_t.txt")

        module_positions = []
        module_intensity = []
        module_variances = []
        module_directions = []
        for i in range(self.number_of_modules):
            observation = individual_RER['module_' + str(i)]
            module_positions.append(observation[:,:self.dimension])
            module_intensity.append(observation[:,self.dimension])
            module_variances.append(observation[:,self.dimension+1:2*self.dimension+1])
            module_directions.append(observation[:,2*self.dimension+1:].reshape([individual_RER['momenta'].shape[0],self.dimension-1, self.dimension]))
        module_positions = np.array(module_positions).transpose(1,0,2)
        module_intensity = np.array(module_intensity).transpose(1, 0)
        module_variances = np.array(module_variances).transpose(1, 0, 2)
        module_directions = np.array(module_directions).transpose(1, 0, 2, 3)

        write_3D_array(np.array(module_positions), output_dir, self.name + "__EstimatedParameters__ModulePositions.txt")
        write_3D_array(module_directions, output_dir, self.name + "__EstimatedParameters__ModuleDirections.txt")
        write_3D_array(module_intensity, output_dir, self.name + "__EstimatedParameters__ModuleIntensities.txt")
        write_3D_array(module_variances, output_dir, self.name + "__EstimatedParameters__ModuleVariances.txt")



        # Momenta covariance.
        write_2D_array(self.get_covariance_momenta_inverse(), output_dir,
                       self.name + "__EstimatedParameters__CovarianceMomentaInverse.txt")

        # Noise variance.
        write_2D_array(np.sqrt(self.get_noise_variance()), output_dir,
                       self.name + "__EstimatedParameters__NoiseStd.txt")

        #write_3D_array(self.fixed_effects['intensity_classes'], output_dir,
        #               self.name + "__EstimatedParameters__IntensityClasses.txt")

        #write_2D_array(self.individual_random_effects['module_intensities'].get_mean(), output_dir,
        #               self.name + "__EstimatedParameters__MeanIntensities.txt")
