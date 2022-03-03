import os
import shutil
import time

from ..core.estimators.gradient_ascent import GradientAscent
from ..core.estimators.scipy_optimize import ScipyOptimize
from ..in_out.array_readers_and_writers import *
from ..in_out.dataset_functions import create_dataset
from ..core import default
from ..core.models.longitudinal_atlas import LongitudinalAtlas


def estimate_longitudinal_registration_for_subject(
        i, template_specifications, dataset_specifications,
        model_options, estimator_options,
        registration_output_path,
        full_subject_ids, full_dataset_filenames, full_visit_ages,
        global_dimension, global_tensor_scalar_type, global_tensor_integer_type, overwrite=True):
    """
    Create the dataset object.
    """

    dataset_specifications['dataset_filenames'] = [full_dataset_filenames[i]]
    dataset_specifications['visit_ages'] = [full_visit_ages[i]]
    dataset_specifications['subject_ids'] = [full_subject_ids[i]]

    dataset = create_dataset(template_specifications,
                             dimension=global_dimension,
                             **dataset_specifications)

    """
    Create a dedicated output folder for the current subject, adapt the global settings.
    """

    subject_registration_output_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__subject_' + full_subject_ids[i])

    if not overwrite and os.path.isdir(subject_registration_output_path):
        return None

    logger.info('')
    logger.info('[ longitudinal registration of subject ' + full_subject_ids[i] + ' ]')
    logger.info('')

    if os.path.isdir(subject_registration_output_path):
        shutil.rmtree(subject_registration_output_path)
        os.mkdir(subject_registration_output_path)

    estimator_options['state_file'] = os.path.join(subject_registration_output_path, 'deformetrica-state.p')

    """
    Create the model object.
    """

    model = LongitudinalAtlas(template_specifications, **model_options)
    individual_RER = model.initialize_random_effects_realization(dataset.number_of_subjects, **model_options)

    # In case of given initial random effect realizations, select only the relevant ones.
    for (random_effect_name) in ['onset_age', 'acceleration', 'sources']:
        if individual_RER[random_effect_name].shape[0] > 1:
            individual_RER[random_effect_name] = np.array([individual_RER[random_effect_name][i]])

    model.initialize_noise_variance(dataset, individual_RER)

    """
    Create the estimator object.
    """

    estimator_options['individual_RER'] = individual_RER

    if estimator_options['optimization_method_type'].lower() == 'GradientAscent'.lower():
        estimator = GradientAscent(model, dataset, output_dir=subject_registration_output_path, **estimator_options)
    elif estimator_options['optimization_method_type'].lower() in ['ScipyLBFGS'.lower(), 'ScipyPowell'.lower()]:
        estimator = ScipyOptimize(model, dataset, output_dir=subject_registration_output_path, **estimator_options)
    else:
        estimator = ScipyOptimize(model, dataset, output_dir=subject_registration_output_path, **estimator_options)

    # estimator(model, dataset, output_dir=subject_registration_output_path, **estimator_options)

    """
    Launch.
    """

    if not os.path.exists(subject_registration_output_path):
        os.makedirs(subject_registration_output_path)

    start_time = 0.0
    end_time = 0.0

    model.name = 'LongitudinalRegistration'
    logger.info('')
    logger.info('[ update method of the ' + estimator.name + ' optimizer ]')

    try:
        start_time = time.time()
        estimator.update()
        model._write_model_parameters(estimator.individual_RER, subject_registration_output_path)
        end_time = time.time()

    except RuntimeError as error:
        logger.info(
            '>> Failure of the longitudinal registration procedure for subject %s: %s' % (full_subject_ids[i], error))

        if not (estimator.name.lower() == 'scipyoptimize' and estimator.method.lower() == 'powell'):
            logger.info('>> Second try with the ScipyPowell optimiser.')

            estimator = ScipyOptimize(
                model, dataset, individual_RER=individual_RER,
                optimization_method_type='ScipyPowell', max_iterations=estimator_options['max_iterations'],
                convergence_tolerance=estimator_options['convergence_tolerance'],
                print_every_n_iters=estimator_options['print_every_n_iters'],
                save_every_n_iters=estimator_options['save_every_n_iters'])

            start_time = time.time()
            estimator.update()
            model._write_model_parameters(estimator.individual_RER, subject_registration_output_path)
            end_time = time.time()

    logger.info('')
    logger.info('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    return model


def estimate_longitudinal_registration(template_specifications, dataset_specifications,
                                       model_options, estimator_options,
                                       output_dir=default.output_dir,
                                       overwrite=True):
    logger.info('')
    logger.info('[ estimate_longitudinal_registration function ]')

    """
    Prepare the loop over each subject.
    """

    registration_output_path = output_dir
    full_dataset_filenames = dataset_specifications['dataset_filenames']
    full_visit_ages = dataset_specifications['visit_ages']
    full_subject_ids = dataset_specifications['subject_ids']
    number_of_subjects = len(full_dataset_filenames)
    estimator_options['save_every_n_iters'] = 100000  # Don't waste time saving intermediate results.

    # Global parameter.
    initial_modulation_matrix_shape = read_2D_array(model_options['initial_modulation_matrix']).shape
    if len(initial_modulation_matrix_shape) > 1:
        global_number_of_sources = initial_modulation_matrix_shape[1]
    else:
        global_number_of_sources = 1

    # Global variables.
    global_dimension = model_options['dimension']
    global_tensor_scalar_type = model_options['tensor_scalar_type']
    global_tensor_integer_type = model_options['tensor_integer_type']

    """
    Launch the individual longitudinal registrations.
    """

    for i in range(number_of_subjects):
        estimate_longitudinal_registration_for_subject(
            i, template_specifications, dataset_specifications,
            model_options, estimator_options,
            registration_output_path,
            full_subject_ids, full_dataset_filenames, full_visit_ages,
            global_dimension, global_tensor_scalar_type, global_tensor_integer_type, overwrite)

    """
    Gather all the individual registration results.
    """

    logger.info('')
    logger.info('[ save the aggregated registration parameters of all subjects ]')
    logger.info('')

    # Gather the individual random effect realizations.
    onset_ages = np.zeros((number_of_subjects,))
    accelerations = np.zeros((number_of_subjects,))
    sources = np.zeros((number_of_subjects, global_number_of_sources))

    for i in range(number_of_subjects):
        subject_registration_output_path = os.path.join(
            registration_output_path, 'LongitudinalRegistration__subject_' + full_subject_ids[i])

        onset_ages[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalRegistration__EstimatedParameters__OnsetAges.txt'))
        accelerations[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalRegistration__EstimatedParameters__Accelerations.txt'))
        sources[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalRegistration__EstimatedParameters__Sources.txt'))

    individual_RER = {}
    individual_RER['sources'] = sources
    individual_RER['onset_age'] = onset_ages
    individual_RER['acceleration'] = accelerations

    # Write temporarily those files.
    temporary_output_path = os.path.join(registration_output_path, 'tmp')
    if os.path.isdir(temporary_output_path):
        shutil.rmtree(temporary_output_path)
    os.mkdir(temporary_output_path)

    path_to_onset_ages = os.path.join(temporary_output_path, 'onset_ages.txt')
    path_to_accelerations = os.path.join(temporary_output_path, 'acceleration.txt')
    path_to_sources = os.path.join(temporary_output_path, 'sources.txt')

    np.savetxt(path_to_onset_ages, onset_ages)
    np.savetxt(path_to_accelerations, accelerations)
    np.savetxt(path_to_sources, sources)

    # Construct the aggregated longitudinal atlas model, and save it.
    dataset_specifications['dataset_filenames'] = full_dataset_filenames
    dataset_specifications['visit_ages'] = full_visit_ages
    dataset_specifications['subject_ids'] = full_subject_ids

    model_options['initial_onset_ages'] = path_to_onset_ages
    model_options['initial_accelerations'] = path_to_accelerations
    model_options['initial_sources'] = path_to_sources

    if not os.path.isdir(registration_output_path):
        os.mkdir(registration_output_path)

    dataset = create_dataset(template_specifications,
                             dimension=global_dimension,
                             **dataset_specifications)

    model = LongitudinalAtlas(template_specifications, **model_options)
    individual_RER = model.initialize_random_effects_realization(dataset.number_of_subjects, **model_options)
    model.initialize_noise_variance(dataset, individual_RER)

    model.name = 'LongitudinalRegistration'
    model.write(dataset, None, individual_RER, registration_output_path)
