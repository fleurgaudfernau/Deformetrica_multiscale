import logging
import os.path
import _pickle as pickle

from ...core import default
from ...core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from ...core.estimators.abstract_estimator import AbstractEstimator
from ...core.estimators.gradient_ascent import GradientAscent
from ...in_out.array_readers_and_writers import *

logger = logging.getLogger(__name__)


class McmcSaem(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method_type='undefined', individual_RER={},
                 max_iterations=default.max_iterations,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 sampler=default.sampler,
                 individual_proposal_distributions=default.individual_proposal_distributions,
                 sample_every_n_mcmc_iters=default.sample_every_n_mcmc_iters,
                 convergence_tolerance=default.convergence_tolerance,
                 callback=None, output_dir=default.output_dir,
                 scale_initial_step_size=default.scale_initial_step_size, initial_step_size=default.initial_step_size,
                 max_line_search_iterations=default.max_line_search_iterations,
                 line_search_shrink=default.line_search_shrink, line_search_expand=default.line_search_expand,
                 load_state_file=default.load_state_file, state_file=default.state_file,
                 **kwargs):

        super().__init__(statistical_model=statistical_model, dataset=dataset, name='McmcSaem',
                         # optimized_log_likelihood=optimized_log_likelihood,
                         max_iterations=max_iterations,
                         convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                         callback=callback, state_file=state_file, output_dir=output_dir)

        assert optimization_method_type.lower() == self.name.lower()

        assert sampler.lower() == 'SrwMhwg'.lower(), \
            "The only available sampler for now is the Symmetric-Random-Walk Metropolis-Hasting-within-Gibbs " \
            "(SrwMhhwg) sampler."
        self.sampler = SrwMhwgSampler(individual_proposal_distributions=individual_proposal_distributions)
        print("individual_proposal_distributions", individual_proposal_distributions)
        self.current_mcmc_iteration = 0
        self.sample_every_n_mcmc_iters = sample_every_n_mcmc_iters
        self.number_of_burn_in_iterations = None  # Number of iterations without memory.
        self.memory_window_size = 1  # Size of the averaging window for the acceptance rates.

        self.number_of_trajectory_points = min(self.max_iterations, 500)
        self.save_model_parameters_every_n_iters = max(1, int(self.max_iterations / float(self.number_of_trajectory_points)))

        # Initialization of the gradient-based optimizer.
        # TODO let the possibility to choose all options (e.g. max_iterations, or ScipyLBFGS optimizer).
        self.gradient_based_estimator = GradientAscent(
            statistical_model, dataset,
            optimized_log_likelihood='class2',
            max_iterations=5, convergence_tolerance=convergence_tolerance,
            print_every_n_iters=1, save_every_n_iters=100000,
            scale_initial_step_size=scale_initial_step_size, initial_step_size=initial_step_size,
            max_line_search_iterations=max_line_search_iterations,
            line_search_shrink=line_search_shrink,
            line_search_expand=line_search_expand,
            output_dir=output_dir, individual_RER=individual_RER,
            optimization_method_type='GradientAscent',
            callback=callback
        )

        self._initialize_number_of_burn_in_iterations()

        # If the load_state_file flag is active, restore context.
        if load_state_file:
            (self.current_iteration, parameters, self.sufficient_statistics, proposal_stds,
             self.current_acceptance_rates, self.average_acceptance_rates,
             self.current_acceptance_rates_in_window, self.average_acceptance_rates_in_window,
             self.model_parameters_trajectory, self.individual_random_effects_samples_stack) = self._load_state_file()
            self._set_parameters(parameters)
            self.sampler.set_proposal_standard_deviations(proposal_stds)
            logger.info("State file loaded, it was at iteration %d." % self.current_iteration)

        else:
            self.current_iteration = 0
            self.sufficient_statistics = None  # Dictionary of numpy arrays.
            self.current_acceptance_rates = {}  # Acceptance rates of the current iteration.
            self.average_acceptance_rates = {}  # Mean acceptance rates, computed over all past iterations.
            self.current_acceptance_rates_in_window = None  # Memory of the last memory_window_size acceptance rates.
            self.average_acceptance_rates_in_window = None  # Moving average of current_acceptance_rates_in_window.
            self.model_parameters_trajectory = None  # Memory of the model parameters along the estimation.
            self.individual_random_effects_samples_stack = None  # Stack of the last individual random effect samples.

            self._initialize_acceptance_rate_information()
            sufficient_statistics = self._initialize_sufficient_statistics()
            self._initialize_model_parameters_trajectory()
            self._initialize_individual_random_effects_samples_stack()

            # Ensures that all the model fixed effects are initialized.
            self.statistical_model.update_fixed_effects(self.dataset, sufficient_statistics)


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Runs the MCMC-SAEM algorithm and updates the statistical model.
        """

        # Print initial console information.
        logger.info('------------------------------------- Iteration: ' + str(
            self.current_iteration) + ' -------------------------------------')
        logger.info('>> MCMC-SAEM algorithm launched for ' + str(self.max_iterations) + ' iterations (' + str(
            self.number_of_burn_in_iterations) + ' iterations of burn-in).')
        self.statistical_model.print(self.individual_RER)

        # Initialization of the average random effects realizations.
        averaged_population_RER = {key: np.zeros(value.shape) for key, value in self.population_RER.items()}
        averaged_individual_RER = {key: np.zeros(value.shape) for key, value in self.individual_RER.items()}

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.callback_ret and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            step = self._compute_step_size()

            # Simulation.
            current_model_terms = None
            for n in range(self.sample_every_n_mcmc_iters):
                self.current_mcmc_iteration += 1

                # Single iteration of the MCMC.
                self.current_acceptance_rates, current_model_terms = self.sampler.sample(
                    self.statistical_model, self.dataset, self.population_RER, self.individual_RER,
                    current_model_terms)

                # Adapt proposal variances.
                self._update_acceptance_rate_information()
                if not (self.current_mcmc_iteration % self.memory_window_size):
                    self.average_acceptance_rates_in_window = {
                        key: np.mean(self.current_acceptance_rates_in_window[key])
                        for key in self.sampler.individual_proposal_distributions.keys()}
                    self.sampler.adapt_proposal_distributions(
                        self.average_acceptance_rates_in_window,
                        self.current_mcmc_iteration,
                        not self.current_iteration % self.print_every_n_iters and n == self.sample_every_n_mcmc_iters - 1)

            # Maximization for the class 1 fixed effects.
            sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
                self.dataset, self.population_RER, self.individual_RER, model_terms=current_model_terms)
            self.sufficient_statistics = {key: value + step * (sufficient_statistics[key] - value) for key, value in
                                          self.sufficient_statistics.items()}
            self.statistical_model.update_fixed_effects(self.dataset, self.sufficient_statistics)

            # Maximization for the class 2 fixed effects.
            fixed_effects_before_maximization = self.statistical_model.get_fixed_effects()
            self._maximize_over_fixed_effects()
            fixed_effects_after_maximization = self.statistical_model.get_fixed_effects()
            fixed_effects = {key: value + step * (fixed_effects_after_maximization[key] - value) for key, value in
                             fixed_effects_before_maximization.items()}
            self.statistical_model.set_fixed_effects(fixed_effects)

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                coefficient_1 = float(self.current_iteration + 1 - self.number_of_burn_in_iterations)
                coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
                averaged_population_RER = {key: value * coefficient_2 + self.population_RER[key] / coefficient_1 for
                                           key, value in averaged_population_RER.items()}
                averaged_individual_RER = {key: value * coefficient_2 + self.individual_RER[key] / coefficient_1 for
                                           key, value in averaged_individual_RER.items()}
                self._update_individual_random_effects_samples_stack()

            else:
                averaged_individual_RER = self.individual_RER
                averaged_population_RER = self.population_RER

            # Saving, printing, writing.
            if not (self.current_iteration % self.save_model_parameters_every_n_iters):
                self._update_model_parameters_trajectory()
            if not (self.current_iteration % self.print_every_n_iters):
                self.print()
            if not (self.current_iteration % self.save_every_n_iters):
                self.write()

        # Finalization -------------------------------------------------------------------------------------------------
        self.population_RER = averaged_population_RER
        self.individual_RER = averaged_individual_RER

    def print(self):
        """
        Prints information.
        """
        # Iteration number.
        logger.info('')
        logger.info('------------------------------------- Iteration: ' + str(
            self.current_iteration) + ' -------------------------------------')

        # Averaged acceptance rates over all the past iterations.
        logger.info('>> Average acceptance rates (all past iterations):')
        for random_effect_name, average_acceptance_rate in self.average_acceptance_rates.items():
            logger.info('\t\t %.2f \t[ %s ]' % (average_acceptance_rate, random_effect_name))

        # Let the model under optimization print information about itself.
        self.statistical_model.print(self.individual_RER)

    def write(self, population_RER=None, individual_RER=None):
        """
        Save the current results.
        """
        # Call the write method of the statistical model.
        if population_RER is None:
            population_RER = self.population_RER
        if individual_RER is None:
            individual_RER = self.individual_RER
        self.statistical_model.write(self.dataset, population_RER, individual_RER, self.output_dir,
                                     update_fixed_effects=False)

        # Save the recorded model parameters trajectory.
        # self.model_parameters_trajectory is a list of dictionaries
        #modif fg : avoid memory error
        """
        np.save(os.path.join(self.output_dir, self.statistical_model.name + '__EstimatedParameters__Trajectory.npy'),
                np.array(
                    {key: value[:(1 + int(self.current_iteration / float(self.save_model_parameters_every_n_iters)))]
                     for key, value in self.model_parameters_trajectory.items()}))

        # Save the memorized individual random effects samples.
        if self.current_iteration > self.number_of_burn_in_iterations:
            np.save(os.path.join(self.output_dir,
                                 self.statistical_model.name + '__EstimatedParameters__IndividualRandomEffectsSamples.npy'),
                    {key: value[:(self.current_iteration - self.number_of_burn_in_iterations)] for key, value in
                     self.individual_random_effects_samples_stack.items()})
         
        """
        # Dump state file.
        #modif fg : fichier très lourd !
        #self._dump_state_file()

    ####################################################################################################################
    ### Private_maximize_over_remaining_fixed_effects() method and associated utilities:
    ####################################################################################################################

    def _maximize_over_fixed_effects(self):
        """
        Update the model fixed effects for which no closed-form update is available (i.e. based on sufficient
        statistics).
        """

        # Default optimizer, if not initialized in the launcher.
        # Should better be done in a dedicated initializing method. TODO.
        if self.statistical_model.has_maximization_procedure is not None \
                and self.statistical_model.has_maximization_procedure:
            self.statistical_model.maximize(self.individual_RER, self.dataset)

        else:
            self.gradient_based_estimator.initialize()

            if self.gradient_based_estimator.verbose > 0:
                logger.info('')
                logger.info('[ maximizing over the fixed effects with the %s optimizer ]'
                            % self.gradient_based_estimator.name)

            success = False
            while not success:
                try:
                    self.gradient_based_estimator.update()
                    success = True
                except RuntimeError as error:
                    logger.info('>> ' + str(error.args[0]) + ' [ in mcmc_saem ]')
                    self.statistical_model.adapt_to_error(error.args[1])

            if self.gradient_based_estimator.verbose > 0:
                logger.info('')
                logger.info('[ end of the gradient-based maximization ]')

        # if self.current_iteration < self.number_of_burn_in_iterations:
        #     self.statistical_model.preoptimize(self.dataset, self.individual_RER)

    ####################################################################################################################
    ### Other private methods:
    ####################################################################################################################

    def _compute_step_size(self):
        aux = self.current_iteration - self.number_of_burn_in_iterations + 1
        if aux <= 0:
            return 1.0
        else:
            return 1.0 / aux

    def _initialize_number_of_burn_in_iterations(self):
        if self.number_of_burn_in_iterations is None:
            # Because some models will set it manually (e.g. deep Riemannian models)
            if self.max_iterations > 4000:
                self.number_of_burn_in_iterations = self.max_iterations - 2000
            else:
                self.number_of_burn_in_iterations = int(self.max_iterations / 2)

    def _initialize_acceptance_rate_information(self):
        # Initialize average_acceptance_rates.
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

        # Initialize current_acceptance_rates_in_window.
        self.current_acceptance_rates_in_window = {key: np.zeros((self.memory_window_size,))
                                                   for key in self.sampler.individual_proposal_distributions.keys()}
        self.average_acceptance_rates_in_window = {key: 0.0
                                                   for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        # Update average_acceptance_rates.
        coefficient_1 = float(self.current_mcmc_iteration)
        coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
        self.average_acceptance_rates = {key: value * coefficient_2 + self.current_acceptance_rates[key] / coefficient_1
                                         for key, value in self.average_acceptance_rates.items()}

        # Update current_acceptance_rates_in_window.
        for key in self.current_acceptance_rates_in_window.keys():
            self.current_acceptance_rates_in_window[key][(self.current_mcmc_iteration - 1) % self.memory_window_size] = \
                self.current_acceptance_rates[key]

    def _initialize_sufficient_statistics(self):
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(self.dataset, self.population_RER,
                                                                                     self.individual_RER)
        self.sufficient_statistics = {key: np.zeros(value.shape) for key, value in sufficient_statistics.items()}
        return sufficient_statistics

    ####################################################################################################################
    ### Model parameters trajectory saving methods:
    ####################################################################################################################

    def _initialize_model_parameters_trajectory(self):
        self.model_parameters_trajectory = {}
        for (key, value) in self.statistical_model.get_fixed_effects(mode='all').items():
            self.model_parameters_trajectory[key] = np.zeros((self.number_of_trajectory_points + 1, value.size))
            self.model_parameters_trajectory[key][0, :] = value.flatten()

    def _update_model_parameters_trajectory(self):
        for (key, value) in self.statistical_model.get_fixed_effects(mode='all').items():
            self.model_parameters_trajectory[key][
            int(self.current_iteration / float(self.save_model_parameters_every_n_iters)), :] = value.flatten()

    def _get_vectorized_individual_RER(self):
        return np.concatenate([value.flatten() for value in self.individual_RER.values()])

    def _initialize_individual_random_effects_samples_stack(self):
        number_of_concentration_iterations = self.max_iterations - self.number_of_burn_in_iterations
        self.individual_random_effects_samples_stack = {}
        for (key, value) in self.individual_RER.items():
            if number_of_concentration_iterations > 0:
                self.individual_random_effects_samples_stack[key] = np.zeros(
                    (number_of_concentration_iterations, value.size))
                self.individual_random_effects_samples_stack[key][0, :] = value.flatten()

    def _update_individual_random_effects_samples_stack(self):
        for (key, value) in self.individual_RER.items():
            self.individual_random_effects_samples_stack[key][
            self.current_iteration - self.number_of_burn_in_iterations - 1, :] = value.flatten()

    ####################################################################################################################
    ### Pickle dump methods.
    ####################################################################################################################

    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.population_RER)
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) \
               + len(self.population_RER) + len(self.individual_RER)
        return out

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return (d['current_iteration'],
                    d['current_parameters'],
                    d['current_sufficient_statistics'],
                    d['current_proposal_stds'],
                    d['current_acceptance_rates'],
                    d['average_acceptance_rates'],
                    d['current_acceptance_rates_in_window'],
                    d['average_acceptance_rates_in_window'],
                    d['trajectory'],
                    d['samples'])

    def _dump_state_file(self):
        d = {
            'current_iteration': self.current_iteration,
            'current_parameters': self._get_parameters(),
            'current_sufficient_statistics': self.sufficient_statistics,
            'current_proposal_stds': self.sampler.get_proposal_standard_deviations(),
            'current_acceptance_rates': self.current_acceptance_rates,
            'average_acceptance_rates': self.average_acceptance_rates,
            'current_acceptance_rates_in_window': self.current_acceptance_rates_in_window,
            'average_acceptance_rates_in_window': self.average_acceptance_rates_in_window,
            'trajectory': self.model_parameters_trajectory,
            'samples': self.individual_random_effects_samples_stack
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(d, f, protocol=4)
