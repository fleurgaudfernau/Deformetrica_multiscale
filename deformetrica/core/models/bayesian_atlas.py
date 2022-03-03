import math

import torch


from ...support import kernels as kernel_factory
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

import logging
logger = logging.getLogger(__name__)


class BayesianAtlas(AbstractStatisticalModel):
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

                 gpu_mode=default.gpu_mode,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='BayesianAtlas', gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode
        self.number_of_processes = number_of_processes

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['covariance_momenta_inverse'] = None
        self.fixed_effects['noise_variance'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points

        self.priors['covariance_momenta'] = InverseWishartDistribution()
        print("self.priors['covariance_momenta'] ", self.priors['covariance_momenta'])
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()
        print("self.priors['noise_variance'] ", self.priors['noise_variance'])

        self.individual_random_effects['momenta'] = NormalDistribution()

        # Deformation.
        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         objects_noise_variance, self.multi_object_attachment) = create_template_metadata(
            template_specifications, self.dimension, gpu_mode=gpu_mode)

        self.template = DeformableMultiObject(object_list)
        # self.template.update()

        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                               self.dimension, self.objects_name)
        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Covariance momenta.
        self.fixed_effects['covariance_momenta_inverse'] = initialize_covariance_momenta_inverse(
            self.fixed_effects['control_points'], self.exponential.kernel, self.dimension)
        self.priors['covariance_momenta'].scale_matrix = np.linalg.inv(self.fixed_effects['covariance_momenta_inverse'])
        print("self.priors['covariance_momenta'] ", self.priors['covariance_momenta'])
	
        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(objects_noise_variance)
        self.objects_noise_variance_prior_normalized_dof = [elt['noise_variance_prior_normalized_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Momenta random effect.
        self.individual_random_effects['momenta'].mean = np.zeros((self.number_of_control_points * self.dimension,))
        self.individual_random_effects['momenta'].set_covariance_inverse(
            self.fixed_effects['covariance_momenta_inverse'])

    def initialize_random_effects_realization(
            self, number_of_subjects,
            initial_momenta=default.initial_momenta,
            covariance_momenta_prior_normalized_dof=default.covariance_momenta_prior_normalized_dof,
            **kwargs):

        # Initialize the random effects realization.
        individual_RER = {
            'momenta': initialize_momenta(initial_momenta, self.number_of_control_points, self.dimension,
                                          number_of_subjects)
        }

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
        template_data, template_points, control_points = self._fixed_effects_to_torch_tensors(False)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, False)

        residuals_per_object = sum(self._compute_residuals(
            dataset, template_data, template_points, control_points, momenta))
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

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self, mode='class2'):
        out = {}

        if mode == 'class2':
            if not self.freeze_template:
                for key, value in self.fixed_effects['template_data'].items():
                    out[key] = value
            if not self.freeze_control_points:
                out['control_points'] = self.fixed_effects['control_points']

        elif mode == 'all':
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
            out['control_points'] = self.fixed_effects['control_points']
            out['covariance_momenta_inverse'] = self.fixed_effects['covariance_momenta_inverse']
            out['noise_variance'] = self.fixed_effects['noise_variance']

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
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
        template_data, template_points, control_points = self._fixed_effects_to_torch_tensors(with_grad)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete')

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        residuals = self._compute_residuals(dataset, template_data, template_points, control_points, momenta)

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
            regularity = self._compute_random_effects_regularity(momenta)
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, control_points)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    if self.use_sobolev_gradient:
                        gradient['landmark_points'] = self.sobolev_kernel.convolve(
                            template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                            template_points['landmark_points'].grad.detach()).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
            if not self.freeze_control_points:
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if mode == 'complete':
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()

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

        # Empirical residuals variances, for each object ---------------------------------------------------------------
        sufficient_statistics['S2'] = np.zeros((self.number_of_objects,))

        # Trick to save useless computations. Could be extended to work in the multi-object case as well ...
        if model_terms is not None and self.number_of_objects == 1:
            sufficient_statistics['S2'][0] += - 2 * np.sum(model_terms) * self.get_noise_variance()
            return sufficient_statistics

        # Standard case.
        if residuals is None:
            template_data, template_points, control_points = self._fixed_effects_to_torch_tensors(False)
            momenta = self._individual_RER_to_torch_tensors(individual_RER, False)
            residuals = self._compute_residuals(dataset, template_data, template_points, control_points, momenta)
            residuals = [torch.sum(residuals_i) for residuals_i in residuals]

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

    def _compute_random_effects_regularity(self, momenta):
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

    def _compute_residuals(self, dataset, template_data, template_points, control_points, momenta):
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
            residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

        return residuals

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
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

        return template_data, template_points, control_points

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Momenta.
        momenta = individual_RER['momenta']
        momenta = torch.from_numpy(momenta).type(self.tensor_scalar_type).requires_grad_(with_grad)
        return momenta

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
        template_data, template_points, control_points = self._fixed_effects_to_torch_tensors(False)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, False)

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

        # Momenta covariance. modif fg
        #write_2D_array(self.get_covariance_momenta_inverse(), output_dir,
        #               self.name + "__EstimatedParameters__CovarianceMomentaInverse.txt")

        # Noise variance.
        write_2D_array(np.sqrt(self.get_noise_variance()), output_dir,
                       self.name + "__EstimatedParameters__NoiseStd.txt")
