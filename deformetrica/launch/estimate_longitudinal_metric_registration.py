import os
import shutil
import time
from multiprocessing import Pool

from ..core.estimators.scipy_optimize import ScipyOptimize
from ..in_out.array_readers_and_writers import *
from ..in_out.dataset_functions import *
from ..launch.estimate_longitudinal_metric_model import instantiate_longitudinal_metric_model


def estimate_longitudinal_registration_for_subject(args):
    i, general_settings, xml_parameters, registration_output_path, \
    full_dataset = args

    Settings().initialize(general_settings)

    logger.info('')
    logger.info('[ longitudinal registration of subject ' + full_dataset.subject_ids[i] + ' ]')
    logger.info('')

    """
    Create the dataset object.
    """

    dataset = create_image_dataset([full_dataset.subject_ids[i] for _ in range(len(full_dataset.times[i]))],
                                    full_dataset.deformable_objects[i],
                                    full_dataset.times[i])

    """
    Create a dedicated output folder for the current subject, adapt the global settings.
    """

    subject_registration_output_path = os.path.join(
        registration_output_path, 'LongitudinalMetricRegistration__subject_' + full_dataset.subject_ids[i])
    if os.path.isdir(subject_registration_output_path):
        shutil.rmtree(subject_registration_output_path)
        os.mkdir(subject_registration_output_path)

    Settings().output_dir = subject_registration_output_path
    Settings().state_file = os.path.join(subject_registration_output_path, 'pydef_state.p')

    """
    Create the model object.
    """
    Settings().number_of_processes = 1

    model, individual_RER = instantiate_longitudinal_metric_model(xml_parameters, dataset, observation_type='image')

    model.is_frozen['v0'] = True
    model.is_frozen['p0'] = True
    model.is_frozen['reference_time'] = True
    model.is_frozen['onset_age_variance'] = True
    model.is_frozen['log_acceleration_variance'] = True
    model.is_frozen['noise_variance'] = True
    model.is_frozen['metric_parameters'] = True
    model.is_frozen['noise_variance'] = True

    # In case of given initial random effect realizations, select only the relevant ones.
    for (xml_parameter, random_effect_name) \
            in zip([xml_parameters.initial_onset_ages,
                    xml_parameters.initial_log_accelerations],
                   ['onset_age', 'log_acceleration']):
        if xml_parameter is not None and len(individual_RER[random_effect_name].shape) > 1:
            individual_RER[random_effect_name] = np.array([individual_RER[random_effect_name][i, :]])

    """
    Create the estimator object.
    """

    if xml_parameters.optimization_method_type == 'ScipyPowell'.lower():
        estimator = ScipyOptimize()
        estimator.method = 'Powell'

    elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize()
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.memory_length = xml_parameters.memory_length

    estimator.max_iterations = xml_parameters.max_iterations
    estimator.convergence_tolerance = xml_parameters.convergence_tolerance

    estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    estimator.save_every_n_iters = xml_parameters.save_every_n_iters

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    """
    Launch.
    """

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalMetricRegistration'
    logger.info('')
    logger.info('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    model._write_model_parameters()
    model._write_model_predictions(dataset, estimator.individual_RER, sample=False)
    model._write_individual_RER(dataset, estimator.individual_RER)
    end_time = time.time()
    logger.info('')
    logger.info('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    return model


def estimate_longitudinal_metric_registration(xml_parameters):
    logger.info('')
    logger.info('[ estimate_longitudinal_registration function ]')

    """
    Prepare the loop over each subject.
    """
    # Here all the parameters should be frozen:

    full_dataset = None
    registration_output_path = Settings().output_dir

    # Two alternatives: scalar dataset or image dataset for now.
    observation_type = 'None'

    template_specifications = xml_parameters.template_specifications
    for val in template_specifications.values():
        if val['deformable_object_type'].lower() == 'scalar':
            full_dataset = read_and_create_scalar_dataset(xml_parameters)
            observation_type = 'scalar'
            break

    if full_dataset is None:
        full_dataset = read_and_create_image_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                                                xml_parameters.subject_ids, xml_parameters.template_specifications)
        observation_type = 'image'

    number_of_subjects = full_dataset.number_of_subjects
    xml_parameters.save_every_n_iters = 100000  # Don't waste time saving intermediate results.

    assert not xml_parameters.initialization_heuristic, "Should not be used for registrations."

    """
    Launch the individual longitudinal registrations.
    """

    # Multi-threaded version.
    if Settings().number_of_processes > 1:
        pool = Pool(processes=Settings().number_of_processes)
        args = [(i, Settings().serialize(), xml_parameters, registration_output_path,
                full_dataset)
                for i in range(number_of_subjects)]
        _ = pool.map(estimate_longitudinal_registration_for_subject, args)[-1]
        pool.close()
        pool.join()

    # Single thread version.
    else:
        for i in range(number_of_subjects):
            _ = estimate_longitudinal_registration_for_subject((
                i, Settings().serialize(), xml_parameters, registration_output_path,
                full_dataset))

    """
    Gather all the individual registration results.
    """

    logger.info('')
    logger.info('[ save the aggregated registration parameters of all subjects ]')
    logger.info('')

    # Gather the individual random effect realizations.
    onset_ages = np.zeros((number_of_subjects,))
    log_accelerations = np.zeros((number_of_subjects,))
    sources = np.zeros((number_of_subjects, xml_parameters.number_of_sources))

    for i in range(number_of_subjects):
        subject_registration_output_path = os.path.join(
            registration_output_path, 'LongitudinalMetricRegistration__subject_' + full_dataset.subject_ids[i])

        onset_ages[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalMetricRegistration_onset_ages.txt'))
        log_accelerations[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalMetricRegistration_log_accelerations.txt'))

        sources[i, :] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalMetricRegistration_sources.txt'))

    individual_RER = {}
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations
    individual_RER['sources'] = sources


    # Write temporarily those files.
    temporary_output_path = os.path.join(registration_output_path, 'tmp')
    if os.path.isdir(temporary_output_path):
        shutil.rmtree(temporary_output_path)
    os.mkdir(temporary_output_path)

    path_to_onset_ages = os.path.join(temporary_output_path, 'onset_ages.txt')
    path_to_log_accelerations = os.path.join(temporary_output_path, 'log_acceleration.txt')

    np.savetxt(path_to_onset_ages, onset_ages)
    np.savetxt(path_to_log_accelerations, log_accelerations)

    # Now using the write method of the model with the whole dataset, with the new log_accelerations and onset_ages
    Settings().output_dir = registration_output_path
    if not os.path.isdir(Settings().output_dir):
        os.mkdir(Settings().output_dir)

    model, _ = instantiate_longitudinal_metric_model(xml_parameters, full_dataset, observation_type='image')
    model.name = 'LongitudinalRegistration'
    model.write(full_dataset, None, individual_RER, sample=False, update_fixed_effects=False)
