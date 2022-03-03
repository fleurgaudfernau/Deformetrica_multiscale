import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

import math

import torch
from copy import deepcopy

from ...in_out.array_readers_and_writers import *
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...support.utilities.general_settings import Settings
from ...support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from ...support.probability_distributions.multi_scalar_normal_distribution import \
    MultiScalarNormalDistribution
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...core.observations.deformable_objects.image import Image

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import logging
logger = logging.getLogger(__name__)


class DeepPga(AbstractStatisticalModel):
    """
    Longitudinal metric learning. Should handle any dimension.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractStatisticalModel.__init__(self)

        self.observation_type = None
        self.name = 'DeepPga'

        self.number_iterations = 0

        self.number_of_subjects = None
        self.number_of_objects = None

        #Whether there is a parallel transport to compute (not in 1D for instance.)
        self.no_parallel_transport = True
        self.latent_space_dimension = None
        self.net = None
        self.has_maximization_procedure = True

        self.template = None

        self.fixed_effects['noise_variance'] = True
        self.fixed_effects['metric_parameters'] = None

        # Dictionary of prior distributions
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        # Dictionary of probability distributions.
        self.individual_random_effects['latent_position'] = MultiScalarNormalDistribution()

        # Dictionary of booleans
        self.is_frozen = {}
        self.is_frozen['noise_variance'] = False
        self.is_frozen['metric_parameters'] = False

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = np.float64(nv)

    def get_metric_parameters(self):
        return self.fixed_effects['metric_parameters']

    def set_metric_parameters(self, metric_parameters):
        self.fixed_effects['metric_parameters'] = metric_parameters

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self, mode='all'):
        out = {}
        if not self.is_frozen['metric_parameters']:
            out['metric_parameters'] = self.fixed_effects['metric_parameters']
        return deepcopy(out)

    def set_fixed_effects(self, fixed_effects):
        if not self.is_frozen['metric_parameters']:
            self.set_metric_parameters(fixed_effects['metric_parameters'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Initializations of prior parameters
        """
        self.initialize_noise_variables()

        for (key, val) in self.is_frozen.items():
            logger.info(key, val)

        if isinstance(self.template, DeformableMultiObject):
            assert isinstance(self.template.object_list[0], Image)
            self.observation_type = 'Image'
            logger.info('Observation type is image')
            self.observation_type = 'image'

        else:
            logger.info('Observation type is numpy')
            self.observation_type = 'numpy'


    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER,
                               mode='complete', with_grad=False, modified_individual_RER='all'):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        metric_parameters = self._fixed_effects_to_torch_tensors(with_grad)
        latent_positions = self._individual_RER_to_torch_tensors(individual_RER, with_grad)

        residuals = self._compute_residuals(dataset, metric_parameters, latent_positions, with_grad=with_grad)

        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER, residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        attachments = - 0.5 * residuals / self.get_noise_variance()
        attachment = torch.sum(attachments)

        regularity = self._compute_random_effects_regularity(latent_positions)

        if mode == 'complete':
            regularity += self._compute_class1_priors_regularity()
            #regularity += self._compute_class2_priors_regularity()

        if with_grad:
            total = attachment + regularity
            total.backward(retain_graph=False)

            # Gradients of the effects with no closed form update.
            gradient = {}
            if not self.is_frozen['metric_parameters']:
                gradient['metric_parameters'] = self.net.get_gradient()
                self.net.zero_grad()

            if mode == 'complete':
                gradient['latent_position'] = latent_positions.grad.detach().cpu().numpy()

            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient
            elif mode == 'model':
                return attachments.detach().cpu().numpy(), gradient

        else:
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy()[0], regularity.detach().cpu().numpy()[0]
            elif mode == 'model':
                return attachments.detach().cpu().numpy()

    def maximize(self, individual_RER, dataset):
        latent_positions = torch.from_numpy(individual_RER['latent_position']).type(Settings().tensor_scalar_type)

        if self.observation_type == 'image':
            data = np.array([elt[0].object_list[0].get_intensities() for elt in dataset.deformable_objects])
        elif self.observation_type == 'numpy':
            data = np.array([elt[0] for elt in dataset.deformable_objects])
        else:
            raise ValueError('Unrecognized observation type')

        data_torch = torch.from_numpy(data).type(Settings().tensor_scalar_type)

        nn_dataset = TensorDataset(latent_positions, data_torch)
        dataloader = DataLoader(nn_dataset, batch_size=15, shuffle=True)
        optimizer = optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=0)
        criterion = nn.MSELoss()

        nb_epochs = 10
        for epoch in range(nb_epochs):
            train_loss = 0
            nb_train_batches = 0

            for (z, y) in dataloader:
                predicted = self.net(z)
                loss = criterion(predicted, y)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().detach().numpy()
                nb_train_batches += 1

            train_loss /= nb_train_batches
            if epoch % 10 == 0:
                logger.info("Epoch {}/{}".format(epoch, nb_epochs),
                      "Train loss:", train_loss)

        self.set_metric_parameters(self.net.get_parameters())

    def _fixed_effects_to_torch_tensors(self, with_grad):
        metric_parameters = torch.from_numpy(self.fixed_effects['metric_parameters']).\
            type(Settings().tensor_scalar_type).requires_grad_((not self.is_frozen['metric_parameters']) and with_grad)

        return metric_parameters

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        latent_positions = torch.from_numpy(individual_RER['latent_position']).requires_grad_(with_grad).type(Settings().tensor_scalar_type)

        return latent_positions

    def _compute_residuals(self, dataset, metric_parameters, latent_positions, with_grad=False):
        targets = dataset.deformable_objects  # A list of list

        number_of_objects = dataset.number_of_subjects

        self.net.set_parameters(metric_parameters)

        residuals = torch.from_numpy(np.zeros((number_of_objects,))).type(Settings().tensor_scalar_type)

        for i in range(number_of_objects):
            assert len(targets[i]) == 1, 'This is not a cross-sectionnal dataset !'
            prediction = self.net(latent_positions[i])
            if self.observation_type == 'image':
                target_torch = torch.from_numpy(targets[i][0].object_list[0].get_intensities())
                # target_torch = targets[i][0].object_list[0].get_intensities_torch()
            else:
                target_torch = torch.from_numpy(targets[i][0])

            residual = (target_torch - prediction)**2

            residuals[i] = torch.sum(residual)
            #
            # if self.observation_type == 'numpy':
            #     residuals[i] = torch.sum(residual)
            #
            # else:
            #     if Settings().dimension == 2:
            #         residuals[i] = torch.sum(torch.sum(residual.view(target_torch.size()), 0), 0)
            #     elif Settings().dimension == 3:
            #         residuals[i] = torch.sum(torch.sum(torch.sum(residual.view(target_torch.size()), 1), 1,), 1)
            #     else:
            #         raise ValueError('Dimension should not be {}'.format(Settings().dimension))


        np.savetxt(os.path.join(Settings().output_dir, 'residuals.txt'), np.array([elt.data.numpy() for elt in residuals]))

        return residuals


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_random_effects_regularity(self, latent_positions):
        """
        Fully torch.
        """
        number_of_objects = latent_positions.shape[0]
        regularity = 0.0

        # Onset age random effect.
        for i in range(number_of_objects):
            regularity += self.individual_random_effects['latent_position'].compute_log_likelihood_torch(latent_positions[i])

        # Noise random effect
        regularity -= 0.5 * number_of_objects * math.log(self.fixed_effects['noise_variance'])

        return regularity

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None, model_terms=None):
        sufficient_statistics = {}

        if residuals is None:
            metric_parameters = self._fixed_effects_to_torch_tensors(False)
            latent_positions = self._individual_RER_to_torch_tensors(individual_RER, False)
            residuals = self._compute_residuals(dataset, metric_parameters, latent_positions, with_grad=False)

        if not self.is_frozen['noise_variance']:
            sufficient_statistics['S1'] = 0.
            for i in range(len(residuals)):
                sufficient_statistics['S1'] += torch.sum(residuals[i]).cpu().detach().numpy()

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        number_of_subjects = dataset.number_of_subjects
        total_number_of_observations = dataset.total_number_of_observations

        # Updating the noise variance
        if not self.is_frozen['noise_variance']:
            prior_scale = self.priors['noise_variance'].scale_scalars[0]
            prior_dof = self.priors['noise_variance'].degrees_of_freedom[0]
            if self.observation_type == 'numpy':
                noise_dimension = Settings().dimension
                logger.info("Noise dimension automatically set to {}".format(Settings().dimension))
            else:
                if Settings().dimension == 2:
                    noise_dimension = 28 * 28
                    logger.info('Noise dimension automatically set to 28x28 ! watchout')
                else:
                    noise_dimension = 64 * 64 * 64
                    logger.info('Noise dimension automatically set to 64x64x64 ! watchout')
            noise_variance = (sufficient_statistics['S1'] + prior_dof * prior_scale) \
                                        / (noise_dimension * total_number_of_observations + prior_dof)
            self.set_noise_variance(noise_variance)

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Noise variance prior
        if not self.is_frozen['noise_variance']:
            regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _get_lsd_observations(self, individual_RER, dataset):
        assert False

    def initialize_noise_variables(self):
        initial_noise_variance = self.get_noise_variance()
        assert initial_noise_variance > 0
        if len(self.priors['noise_variance'].scale_scalars) == 0:
                self.priors['noise_variance'].scale_scalars.append(0.01 * initial_noise_variance)
        logger.info('Default dof for nosie variance to 200')
        self.priors['noise_variance'].degrees_of_freedom.append(200)


    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, sample=False, update_fixed_effects=False):
        self._write_model_predictions(dataset, individual_RER, sample=sample)
        self._write_model_parameters()
        self._write_lsd_coordinates(individual_RER)
        self.write_sources()

    def _write_model_parameters(self):
        np.savetxt(os.path.join(Settings().output_dir, self.name + '_metric_parameters.txt'),
                   self.get_metric_parameters())

    def _write_lsd_coordinates(self, individual_RER):
        np.savetxt(os.path.join(Settings().output_dir, self.name + '_latent_position.txt'),
                   individual_RER['latent_position'])

    def write_sources(self):
        if self.observation_type == 'image':

            metric_parameters = self._fixed_effects_to_torch_tensors(with_grad=False)

            self.net.set_parameters(metric_parameters)
            for i in range(self.latent_space_dimension):
                direction = np.zeros((self.latent_space_dimension,))
                direction[i] = 1.
                times = np.linspace(-1., 1., 10)
                for t in times:
                    latent_position = direction * t
                    l_p_torch = torch.from_numpy(latent_position).type(Settings().tensor_scalar_type)
                    img = self.net(l_p_torch)
                    self.template.write(['sources_' + str(i) + '_' + str(t) + '.png'], {'image_intensities':img.detach().numpy()})

    def _write_model_predictions(self, dataset, individual_RER, sample=False):

        # We write the targets and the reconstruced images, in png format.
        targets = dataset.deformable_objects  # A list of list
        latent_positions = self._individual_RER_to_torch_tensors(individual_RER, with_grad=False)
        metric_parameters = self._fixed_effects_to_torch_tensors(with_grad=False)

        number_of_objects = dataset.number_of_subjects

        self.net.set_parameters(metric_parameters)

        if self.observation_type == 'image':
            for i in range(number_of_objects):
                assert len(targets[i]) == 1, 'This is not a cross-sectionnal dataset !'
                prediction = self.net(latent_positions[i])
                targets[i][0].write(['target_'+str(i) + '.png'])
                self.template.write(['reconstructed_'+str(i)+'.png'], {'image_intensities': prediction.detach().numpy()})

        elif self.observation_type == 'numpy':
            if self.number_iterations == 0:
                observations = np.array(dataset.deformable_objects).reshape(dataset.number_of_subjects, Settings().dimension)
                np.save(os.path.join(Settings().output_dir, 'observations.npy'), observations)
            reconstructed_observations = self.net(latent_positions).detach().numpy()
            np.save(os.path.join(Settings().output_dir, 'reconstructed_observations_' + str(self.number_iterations) + '.npy'), reconstructed_observations)

        else:
            raise ValueError('Unrecognized observation type')

        self.number_iterations += 1

    def print(self, individual_RER):
        logger.info('>> Model parameters:')
        # Noise variance.
        msg = '\t\t noise_variance    ='
        noise_variance = self.get_noise_variance()
        msg += '\t%.4f\t ; ' % (math.sqrt(noise_variance))
        logger.info(msg[:-4])

