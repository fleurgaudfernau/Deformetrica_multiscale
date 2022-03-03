import logging
logger = logging.getLogger(__name__)

import math

import numpy as np

from core import default
from support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution


class SrwMhwgSamplerForClustering:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self,
                 individual_proposal_distributions=default.individual_proposal_distributions,
                 acceptance_rates_target=30.0):

        # Dictionary of probability distributions.
        self.population_proposal_distributions = {}
        self.individual_proposal_distributions = individual_proposal_distributions

        self.acceptance_rates_target = acceptance_rates_target  # Percentage.

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self, statistical_model, dataset, population_RER, individual_RER, iteration, current_model_terms=None):

        # Initialization -----------------------------------------------------------------------------------------------

        # Initialization of the memory of the current model terms.
        # The contribution of each subject is stored independently.
        if current_model_terms is None:
            current_model_terms = self._compute_model_log_likelihood(statistical_model, dataset,
                                                                     population_RER, individual_RER)

        # Acceptance rate metrics initialization.
        acceptance_rates = {key: 0.0 for key in self.individual_proposal_distributions.keys()}

        # Main loop ----------------------------------------------------------------------------------------------------
        for random_effect_name, proposal_RED in self.individual_proposal_distributions.items():

            # RED: random effect distribution.
            if random_effect_name in ['accelerations', 'onset_ages']:
                model_RED = statistical_model.individual_random_effects['time_parameters']
            else:
                model_RED = statistical_model.individual_random_effects[random_effect_name]

            # Initialize subject lists.
            current_regularity_terms = []
            candidate_regularity_terms = []
            current_RER = []
            candidate_RER = []

            # Shape parameters of the current random effect realization.
            shape_parameters = individual_RER[random_effect_name][0].shape
            modified_subject = np.ones(dataset.number_of_subjects)

            for i in range(dataset.number_of_subjects):
                # Evaluate the current part.
                if random_effect_name in ['accelerations', 'onset_ages']:
                    time_parameters = np.append(individual_RER['accelerations'][i],individual_RER['onset_ages'][i])
                    current_regularity_terms.append(model_RED[individual_RER['classes'][i]].compute_log_likelihood(time_parameters))
                else:
                    current_regularity_terms.append(model_RED.compute_log_likelihood(individual_RER[random_effect_name][i]))
                current_RER.append(individual_RER[random_effect_name][i].flatten())

                # Draw the candidate.
                proposal_RED.mean = current_RER[i]
                candidate_RER.append(proposal_RED.sample())

                if random_effect_name == 'accelerations':
                    for k in range(int(statistical_model.nb_max_component)):
                        if k >= statistical_model.nb_component[individual_RER['classes'][i]]:
                            candidate_RER[i][k] = current_RER[i][k]
                        elif k > 0:
                            if statistical_model.num_component[individual_RER['classes'][i]][k] == \
                                    statistical_model.num_component[individual_RER['classes'][i]][k - 1]:
                                candidate_RER[i][k] = candidate_RER[i][k - 1]

                if random_effect_name == 'classes':
                    if candidate_RER[i] == current_RER[i]: modified_subject[i] = 0

                # Evaluate the candidate part.
                individual_RER[random_effect_name][i] = candidate_RER[i].reshape(shape_parameters)
                if random_effect_name == 'accelerations':
                    time_parameters = np.append(candidate_RER[i], individual_RER['onset_ages'][i])
                    candidate_regularity_terms.append(model_RED[individual_RER['classes'][i]].compute_log_likelihood(time_parameters))
                elif random_effect_name == 'onset_ages':
                    time_parameters = np.append(individual_RER['accelerations'][i], candidate_RER[i])
                    candidate_regularity_terms.append(model_RED[individual_RER['classes'][i]].compute_log_likelihood(time_parameters))
                else:
                    candidate_regularity_terms.append(model_RED.compute_log_likelihood(candidate_RER[i]))

            # Evaluate the candidate terms for all subjects at once, since all contributions are independent.
            candidate_model_terms = self._compute_model_log_likelihood(
                statistical_model, dataset, population_RER, individual_RER, modified_individual_RER=random_effect_name)

            tau = []
            for i in range(dataset.number_of_subjects):

                # logger.info("Sampling summary", random_effect_name,
                #         "attachments:", candidate_model_terms[i], current_model_terms[i],
                #         "regularities:", candidate_regularity_terms[i], current_regularity_terms[i])

                # Acceptance rate.
                tau.append(candidate_model_terms[i] + candidate_regularity_terms[i] \
                           - current_model_terms[i] - current_regularity_terms[i])

            tau = np.array(tau)
            if random_effect_name == 'classes' and iteration % 10 < 5:
                T = 1 - ((5 - (iteration % 10))/5) + 2*np.median(abs(tau[modified_subject != 0])) / (iteration // 20 + 1) * ((5 - (iteration % 10))/5)
            else: T = 1

            if T != 0: tau /= T
            for i in range(dataset.number_of_subjects):
                if modified_subject[i]:
                    # Reject.
                    if math.log(np.random.uniform()) > tau[i] or math.isnan(tau[i]):
                        individual_RER[random_effect_name][i] = current_RER[i].reshape(shape_parameters)

                    # Accept.
                    else:
                        current_model_terms[i] = candidate_model_terms[i]
                        current_regularity_terms[i] = candidate_regularity_terms[i]
                        acceptance_rates[random_effect_name] += 1.0

            # Acceptance rate final scaling for the considered random effect.
            acceptance_rates[random_effect_name] *= 100.0 / float(dataset.number_of_subjects)

        #if iteration == 0: individual_RER['classes'] = np.array([0,1]*25)
        return acceptance_rates, current_model_terms

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    def _compute_model_log_likelihood(self, statistical_model, dataset, population_RER, individual_RER,
                                      modified_individual_RER='all'):
        try:
            return statistical_model.compute_log_likelihood(
                dataset, population_RER, individual_RER, mode='model', modified_individual_RER=modified_individual_RER)

        except ValueError as error:
            logger.info('>> ' + str(error) + ' \t[ in srw_mhwg_sampler ]')
            statistical_model.clear_memory()
            return np.zeros((dataset.number_of_subjects,)) - float('inf')

    def adapt_proposal_distributions(self, current_acceptance_rates_in_window, iteration_number, verbose):
        goal = self.acceptance_rates_target
        msg = '>> Proposal std re-evaluated from:\n'

        for random_effect_name, proposal_distribution in self.individual_proposal_distributions.items():
            ar = current_acceptance_rates_in_window[random_effect_name]
            std = proposal_distribution.get_variance_sqrt()
            msg += '\t\t %.3f ' % std

            if ar > self.acceptance_rates_target:
                std *= 1 + (ar - goal) / ((100 - goal) * math.sqrt(iteration_number + 1))
            else:
                std *= 1 - (goal - ar) / (goal * math.sqrt(iteration_number + 1))

            msg += '\tto\t%.3f \t[ %s ]\n' % (std, random_effect_name)
            proposal_distribution.set_variance_sqrt(std)

        if verbose > 0: logger.info(msg[:-1])

    ####################################################################################################################
    ### Pickle dump methods.
    ####################################################################################################################

    def get_proposal_standard_deviations(self):
        out = {}
        for random_effect_name, proposal_distribution in self.individual_proposal_distributions.items():
            out[random_effect_name] = proposal_distribution.get_variance_sqrt()
        return out

    def set_proposal_standard_deviations(self, stds):
        for random_effect_name, std in stds.items():
            self.individual_proposal_distributions[random_effect_name].set_variance_sqrt(std)