import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import shutil
import xml.etree.ElementTree as et

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import Settings
from src.in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from pydeformetrica.src.launch.estimate_longitudinal_metric_model import estimate_longitudinal_metric_model
from sklearn import datasets, linear_model
from pydeformetrica.src.in_out.dataset_functions import read_and_create_scalar_dataset, read_and_create_image_dataset
from sklearn.decomposition import PCA
from pydeformetrica.src.core.model_tools.manifolds.metric_learning_nets import ScalarNet, ImageNet2d, ImageNet3d, ImageNet2d128
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
from torch import nn

def _initialize_modulation_matrix_and_sources(dataset, p0, v0, number_of_sources):
    unit_v0 = v0/np.linalg.norm(v0)
    unit_v0 = unit_v0.flatten()
    flat_p0 = p0.flatten()
    vectors = []
    for elt in dataset.deformable_objects:
        for e in elt: #To make it lighter in memory, and faster
            e_np = e.cpu().data.numpy()
            dimension = e_np.shape
            e_np = e_np.flatten()
            vector_projected = e_np - np.dot(e_np, unit_v0) * unit_v0
            vectors.append(vector_projected - flat_p0)

    logger.info("Performing principal component analysis on the orthogonal variations, for initialization of A and s_i.")

    # We now do a pca on those vectors
    pca = PCA(n_components=number_of_sources)
    pca.fit(vectors)
    if len(dimension) == 1:
        out = np.transpose(pca.components_)
    else:
        out = np.transpose(pca.components_).reshape((-1,) + dimension)
    for i in range(number_of_sources):
        out[:, i] /= np.linalg.norm(out[:, i])

    sources = []
    for elt in dataset.deformable_objects:
        obs_for_subject = np.array([(im.cpu().data.numpy() - p0).flatten() for im in elt])
        # We average the coordinate of these obs in pca space
        sources.append(np.mean(pca.transform(obs_for_subject), 0))

    return out, sources

def _smart_initialization_individual_effects(dataset):
    """
    least_square regression for each subject, so that yi = ai * t + bi
    output is the list of ais and bis
    this proceeds as if the initialization for the geodesic is a straight line
    """
    logger.info("Performing initial least square regressions on the subjects, for initialization purposes.")

    number_of_subjects = dataset.number_of_subjects
    dimension = dataset.deformable_objects[0][0].cpu().data.numpy().shape

    ais = []
    bis = []

    for i in range(number_of_subjects):

        # Special case of a single observation for the subject
        if len(dataset.times[i]) <= 1:
            ais.append(1.)
            bis.append(0.)

        else:

            least_squares = linear_model.LinearRegression()
            data_for_subject = np.array([elt.cpu().data.numpy().flatten() for elt in dataset.deformable_objects[i]])
            least_squares.fit(dataset.times[i].reshape(-1, 1), data_for_subject)

            a = least_squares.coef_.reshape(dimension)
            if len(a) == 1 and a[0] < 0.001:
                a = np.array([0.001])
            ais.append(a)
            bis.append(least_squares.intercept_.reshape(dimension))

    return ais, bis

def _smart_initialization(dataset, number_of_sources, observation_type):

    observation_times = []
    for times in dataset.times:
        for t in times:
            observation_times.append(t)
    std_obs = np.std(observation_times)

    dataset_reformated = dataset
    if observation_type == 'image':
        dataset_data = []
        for elt in dataset_reformated.deformable_objects:
            subject_data = []
            for im in elt:

                # subject_data.append(im.get_intensities_torch())
                subject_data.append(torch.from_numpy(im.get_intensities()))
            dataset_data.append(subject_data)

        dataset_reformated.deformable_objects = dataset_data

    ais, bis = _smart_initialization_individual_effects(dataset)
    reference_time = np.mean([np.mean(times_i) for times_i in dataset.times])
    v0 = np.mean(ais, 0)

    p0 = 0
    for i in range(dataset.number_of_subjects):
        aux = np.mean(np.array([elt.cpu().data.numpy() for elt in dataset.deformable_objects[i]]), 0)
        p0 += aux
    p0 /= dataset.number_of_subjects

    alphas = []
    onset_ages = []
    for i in range(len(ais)):
        alpha_proposal = np.dot(ais[i].flatten(), v0.flatten())/np.sum(v0**2)
        alpha = max(0.003, min(10., alpha_proposal))
        alphas.append(alpha)

        onset_age_proposal = 1. / alpha * np.dot(p0.flatten() - bis[i].flatten(), v0.flatten())/np.sum(v0**2)
        #onset_age_proposal = np.linalg.norm(p0-bis[i])/np.linalg.norm(ais[i])
        onset_age = max(reference_time - 2 * std_obs, min(reference_time + 2 * std_obs, onset_age_proposal))
        logger.info(onset_age_proposal, onset_age)
        onset_ages.append(onset_age)


    # ADD a normalization step (0 mean, unit variance):
    if True:
        log_accelerations = np.log(alphas)
        log_accelerations = 0.5*(log_accelerations - np.mean(log_accelerations, 0))/np.std(log_accelerations, 0)
        alphas = np.exp(log_accelerations)
        # We want the onset ages to have an std equal to the std of the obser times


        onset_ages = (onset_ages - np.mean(onset_ages, 0))/np.std(onset_ages, 0) * std_obs + np.mean(onset_ages)
        logger.info('std onset_ages vs obs times', np.std(onset_ages), std_obs)

    reference_time = np.mean(onset_ages, 0)

    if number_of_sources > 0:
        modulation_matrix, sources = _initialize_modulation_matrix_and_sources(dataset, p0, v0, number_of_sources)

    else:
        modulation_matrix = None
        sources = None

    if True and sources is not None:
        sources = np.array(sources)
        sources = (sources - np.mean(sources, 0)) / np.std(sources, 0)

    return reference_time, v0, p0, np.array(onset_ages), np.array(alphas), modulation_matrix, sources


if __name__ == '__main__':

    logger.info('')
    logger.info('##############################')
    logger.info('##### PyDeformetrica 1.0 #####')
    logger.info('##############################')

    logger.info('')

    assert len(sys.argv) == 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> "

    model_xml_path = sys.argv[1]
    dataset_xml_path = sys.argv[2]
    optimization_parameters_xml_path = sys.argv[3]

    preprocessings_folder = Settings().preprocessing_dir
    if not os.path.isdir(preprocessings_folder):
        os.mkdir(preprocessings_folder)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)
    xml_parameters._further_initialization()

    """
    1) Simple heuristic for initializing everything but the sources and the modulation matrix.
    """

    smart_initialization_output_path = os.path.join(preprocessings_folder, '1_smart_initialization')
    Settings().output_dir = smart_initialization_output_path

    if not os.path.isdir(smart_initialization_output_path):
        os.mkdir(smart_initialization_output_path)

    # Creating the dataset object
    observation_type = None
    dataset = None

    template_specifications = xml_parameters.template_specifications
    for val in template_specifications.values():
        if val['deformable_object_type'].lower() == 'scalar':
            dataset = read_and_create_scalar_dataset(xml_parameters)
            observation_type = 'scalar'
            break

    if dataset is None:
        dataset = read_and_create_image_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                                                xml_parameters.subject_ids, xml_parameters.template_specifications)
        observation_type = 'image'


    # Heuristic for the initializations
    if xml_parameters.number_of_sources is None or xml_parameters.number_of_sources == 0:
        reference_time, average_a, p0, onset_ages, alphas, modulation_matrix, sources = _smart_initialization(dataset, 0, observation_type)
    else:
        reference_time, average_a, p0, onset_ages, alphas, modulation_matrix, sources = _smart_initialization(dataset, xml_parameters.number_of_sources, observation_type)

    # We save the onset ages and alphas.
    # We then set the right path in the xml_parameters, for the proper initialization.
    write_2D_array(np.log(alphas), "SmartInitialization_log_accelerations.txt")
    xml_parameters.initial_log_accelerations = os.path.join(smart_initialization_output_path, "SmartInitialization_log_accelerations.txt")

    write_2D_array(onset_ages, "SmartInitialization_onset_ages.txt")
    xml_parameters.initial_onset_ages = os.path.join(smart_initialization_output_path, "SmartInitialization_onset_ages.txt")

    if xml_parameters.exponential_type != 'deep':
        write_2D_array(np.array([p0]), "SmartInitialization_p0.txt")
        xml_parameters.p0 = os.path.join(smart_initialization_output_path, "SmartInitialization_p0.txt")

        write_2D_array(np.array([average_a]), "SmartInitialization_v0.txt")
        xml_parameters.v0 = os.path.join(smart_initialization_output_path, "SmartInitialization_v0.txt")

    if modulation_matrix is not None:
        assert sources is not None
        if xml_parameters.exponential_type != 'deep':
            write_2D_array(modulation_matrix, "SmartInitialization_modulation_matrix.txt")
            xml_parameters.initial_modulation_matrix = os.path.join(smart_initialization_output_path, "SmartInitialization_modulation_matrix.txt")
        write_2D_array(sources, "SmartInitialization_sources.txt")
        xml_parameters.initial_sources = os.path.join(smart_initialization_output_path,
                                                          "SmartInitialization_sources.txt")

    xml_parameters.t0 = reference_time

    # Now the stds:
    xml_parameters.initial_log_acceleration_variance = np.var(np.log(alphas))
    xml_parameters.initial_time_shift_variance = np.var(onset_ages)

    if xml_parameters.exponential_type != 'deep':

        assert False, 'The metric model without deep probably does not work now ! to be checked.'

        """
        2) Gradient descent on the mode
        """

        mode_descent_output_path = os.path.join(preprocessings_folder, '2_gradient_descent_on_the_mode')
        # To perform this gradient descent, we use the iniialization heuristic, starting from
        # a flat metric and linear regressions one each subject

        xml_parameters.optimization_method_type = 'GradientAscent'.lower()
        xml_parameters.scale_initial_step_size = True
        xml_parameters.max_iterations = 20
        xml_parameters.save_every_n_iters = 5

        # Freezing some variances !
        xml_parameters.freeze_log_acceleration_variance = True
        xml_parameters.freeze_noise_variance = True
        xml_parameters.freeze_onset_age_variance = True

        # Freezing other variables
        xml_parameters.freeze_modulation_matrix = True
        xml_parameters.freeze_p0 = True

        xml_parameters.output_dir = mode_descent_output_path
        Settings().set_output_dir(mode_descent_output_path)

        logger.info(" >>> Performing gradient descent on the mode.")

        estimate_longitudinal_metric_model(xml_parameters)

        """"""""""""""""""""""""""""""""
        """Creating a xml file"""
        """"""""""""""""""""""""""""""""

        model_xml = et.Element('data-set')
        model_xml.set('deformetrica-min-version', "3.0.0")

        model_type = et.SubElement(model_xml, 'model-type')
        model_type.text = "LongitudinalMetricLearning"

        dimension = et.SubElement(model_xml, 'dimension')
        dimension.text=str(Settings().dimension)

        estimated_alphas = np.loadtxt(os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_alphas.txt'))
        estimated_onset_ages = np.loadtxt(os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_onset_ages.txt'))

        initial_time_shift_std = et.SubElement(model_xml, 'initial-time-shift-std')
        initial_time_shift_std.text = str(np.std(estimated_onset_ages))

        initial_log_acceleration_std = et.SubElement(model_xml, 'initial-log-acceleration-std')
        initial_log_acceleration_std.text = str(np.std(np.log(estimated_alphas)))

        deformation_parameters = et.SubElement(model_xml, 'deformation-parameters')

        exponential_type = et.SubElement(deformation_parameters, 'exponential-type')
        exponential_type.text = xml_parameters.exponential_type

        if xml_parameters.exponential_type == 'parametric':
            interpolation_points = et.SubElement(deformation_parameters, 'interpolation-points-file')
            interpolation_points.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_interpolation_points.txt')
            kernel_width = et.SubElement(deformation_parameters, 'kernel-width')
            kernel_width.text = str(xml_parameters.deformation_kernel_width)

        concentration_of_timepoints = et.SubElement(deformation_parameters,
                                                    'concentration-of-timepoints')
        concentration_of_timepoints.text = str(xml_parameters.concentration_of_time_points)

        estimated_fixed_effects = np.load(os.path.join(mode_descent_output_path,
                                                       'LongitudinalMetricModel_all_fixed_effects.npy'))[
            ()]

        if xml_parameters.exponential_type in ['parametric']: # otherwise it's not saved !
            metric_parameters_file = et.SubElement(deformation_parameters,
                                                        'metric-parameters-file')
            metric_parameters_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_metric_parameters.txt')

        if xml_parameters.number_of_sources is not None and xml_parameters.number_of_sources > 0:
            initial_sources_file = et.SubElement(model_xml, 'initial-sources')
            initial_sources_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_sources.txt')
            number_of_sources = et.SubElement(deformation_parameters, 'number-of-sources')
            number_of_sources.text = str(xml_parameters.number_of_sources)
            initial_modulation_matrix_file = et.SubElement(model_xml, 'initial-modulation-matrix')
            initial_modulation_matrix_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_modulation_matrix.txt')

        t0 = et.SubElement(deformation_parameters, 't0')
        t0.text = str(estimated_fixed_effects['reference_time'])

        v0 = et.SubElement(deformation_parameters, 'v0')
        v0.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_v0.txt')

        p0 = et.SubElement(deformation_parameters, 'p0')
        p0.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_p0.txt')

        initial_onset_ages = et.SubElement(model_xml, 'initial-onset-ages')
        initial_onset_ages.text = os.path.join(mode_descent_output_path,
                                               "LongitudinalMetricModel_onset_ages.txt")

        initial_log_accelerations = et.SubElement(model_xml, 'initial-log-accelerations')
        initial_log_accelerations.text = os.path.join(mode_descent_output_path,
                                                      "LongitudinalMetricModel_log_accelerations.txt")


        model_xml_path = 'model_after_initialization.xml'
        doc = parseString((et.tostring(model_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    else:
        """ 
        What we do : we initialize the basis reference frame. We construct the set of pairs (x_i, y_i) for the nn and 
        train it on these, by batch using adam.
        """

        deep_net_initialization_path = os.path.join(preprocessings_folder, '2_initialize_deep_network')
        Settings().output_dir = deep_net_initialization_path
        if not os.path.isdir(deep_net_initialization_path):
            os.mkdir(deep_net_initialization_path)

        lsd = xml_parameters.latent_space_dimension

        # We need to initialize v0, p0,
        tmin = float('inf')
        tmax = - float('inf')
        for i,elt in enumerate(dataset.times):
            for j,t in enumerate(elt):
                abs_time = alphas[i] * (t - onset_ages[i]) + reference_time
                if abs_time < tmin:
                    tmin = abs_time
                elif abs_time > tmax:
                    tmax = abs_time

        p0 = np.zeros((lsd,)) #0.5 * np.ones((lsd,))
        p0[0] = -1 + 2 * (reference_time - tmin)/(tmax - tmin)

        v0 = np.zeros((lsd,))
        v0[0] = 1.
        v0 /= (tmax - tmin)

        logger.info("Reference time", reference_time, "tmin", tmin, "tmax", tmax, "v0", v0, "p0", p0)

        np.savetxt(os.path.join(deep_net_initialization_path, "p0.txt"), p0)
        np.savetxt(os.path.join(deep_net_initialization_path, "v0.txt"), v0)

        # We rescale the sources:
        # sources /= 2*np.max(np.abs(sources), axis=0)

        np.savetxt(os.path.join(deep_net_initialization_path, "sources.txt"), sources)

        assert lsd == xml_parameters.number_of_sources + 1, "Set lsd correctly"
        modulation_matrix = np.zeros((lsd, xml_parameters.number_of_sources))

        # We create an orthonormal basis to v0 !
        for j in range(xml_parameters.number_of_sources):
            modulation_matrix[j+1, j] = 1.

        write_2D_array(modulation_matrix, "modulation_matrix.txt")

        # We now create the latent space observations:
        lsd_observations = []
        observations = []
        for i, elt in enumerate(dataset.times):
            for j, t in enumerate(elt):
                abs_time = alphas[i] * (t - onset_ages[i]) + reference_time
                lsd_obs = v0 * (abs_time - reference_time) + p0 + np.matmul(modulation_matrix, sources[i])
                lsd_observations.append(lsd_obs)
                observations.append(dataset.deformable_objects[i][j].cpu().data.numpy())

        observations = np.array(observations)
        lsd_observations = np.array(lsd_observations)

        logger.info("Bounding box for lsd_observations", np.max(np.abs(lsd_observations), 0))

        # for elt in lsd_observations:
        #     logger.info(elt)

        lsd_observations = torch.from_numpy(lsd_observations).type(Settings().tensor_scalar_type)
        observations = torch.from_numpy(observations).type(Settings().tensor_scalar_type)

        train_len = int(0.9 * len(lsd_observations))

        train_dataset = TensorDataset(lsd_observations[:train_len], observations[:train_len])
        test_dataset = TensorDataset(lsd_observations[train_len:], observations[train_len:])

        train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # We now fit the neural network and saves the final parameters.

        if observation_type == 'scalar':
            net = ScalarNet(in_dimension=lsd, out_dimension=Settings().dimension)

        elif observation_type == 'image':
            length, width = observations[0].shape
            if Settings().dimension == 2:
                if length == 64:
                    net = ImageNet2d(in_dimension=lsd)
                else:
                    net = ImageNet2d128(in_dimension=lsd)
            else:
                net = ImageNet3d(in_dimension=lsd)

        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)

        test_losses = []

        criterion = nn.MSELoss()
        nb_epochs = 200
        for epoch in range(nb_epochs):
            train_loss = 0
            test_loss = 0
            nb_train_batches = 0
            for (z, y) in train_dataloader:
                nb_train_batches += 1
                var_z = Variable(z)
                var_y = Variable(y)
                predicted = net(var_z)
                loss = criterion(predicted, var_y)
                net.zero_grad()
                loss.backward()
                train_loss += loss.cpu().data.numpy()[0]
                optimizer.step()

            train_loss /= nb_train_batches

            for (z, y) in test_dataloader:
                predicted = net(Variable(z))
                loss = criterion(predicted, Variable(y))
                test_loss = loss.cpu().data.numpy()[0]

            test_losses.append(test_loss)

            if epoch > 5:
                b = False
                for i in range(4):
                    b = b or (test_losses[-i] >= test_losses[-i+1])
                if test_losses[-1] > 1.5 * train_loss:
                    b = False
                if not b:
                    logger.info("Test loss stopped improving, we stop.")
                    break

            logger.info("Epoch {}/{}".format(epoch, nb_epochs),
                  "Train loss:", train_loss,
                  "Test loss:", test_loss)

        metric_parameters = net.get_parameters()
        write_2D_array(metric_parameters, "metric_parameters.txt")

        model_xml = et.Element('data-set')
        model_xml.set('deformetrica-min-version', "3.0.0")

        model_type = et.SubElement(model_xml, 'model-type')
        model_type.text = "LongitudinalMetricLearning"

        # Template information (single template)
        template = et.SubElement(model_xml, 'template')
        for key in xml_parameters.template_specifications.keys():
            obj = et.SubElement(template, 'object')
            obj.set('id', key)
            def_type = et.SubElement(obj, 'deformable-object-type')
            def_type.text = xml_parameters.template_specifications[key]['deformable_object_type']

        dimension = et.SubElement(model_xml, 'dimension')
        dimension.text = str(Settings().dimension)

        latent_space_dimension = et.SubElement(model_xml, 'latent-space-dimension')
        latent_space_dimension.text = str(lsd)

        initial_time_shift_std = et.SubElement(model_xml, 'initial-time-shift-std')
        initial_time_shift_std.text = str(np.std(onset_ages))

        initial_log_acceleration_std = et.SubElement(model_xml, 'initial-log-acceleration-std')
        initial_log_acceleration_std.text = str(np.std(np.log(alphas)))

        initial_noise_std = et.SubElement(model_xml, 'initial-noise-std')
        initial_noise_std.text = str(np.sqrt(test_loss))

        deformation_parameters = et.SubElement(model_xml, 'deformation-parameters')

        exponential_type = et.SubElement(deformation_parameters, 'exponential-type')
        exponential_type.text = xml_parameters.exponential_type

        concentration_of_timepoints = et.SubElement(deformation_parameters,
                                                    'concentration-of-timepoints')
        concentration_of_timepoints.text = str(xml_parameters.concentration_of_time_points)

        metric_parameters_file = et.SubElement(deformation_parameters,
                                               'metric-parameters-file')
        metric_parameters_file.text = os.path.join(deep_net_initialization_path,
                                                   'metric_parameters.txt')

        if xml_parameters.number_of_sources is not None and xml_parameters.number_of_sources > 0:
            initial_sources_file = et.SubElement(model_xml, 'initial-sources')
            initial_sources_file.text = os.path.join(deep_net_initialization_path, 'sources.txt')
            number_of_sources = et.SubElement(deformation_parameters, 'number-of-sources')
            number_of_sources.text = str(xml_parameters.number_of_sources)
            initial_modulation_matrix_file = et.SubElement(model_xml, 'initial-modulation-matrix')
            initial_modulation_matrix_file.text = os.path.join(deep_net_initialization_path,
                                                               'modulation_matrix.txt')

        t0 = et.SubElement(deformation_parameters, 't0')
        t0.text = str(reference_time)

        v0 = et.SubElement(deformation_parameters, 'v0')
        v0.text = os.path.join(deep_net_initialization_path, 'v0.txt')

        p0 = et.SubElement(deformation_parameters, 'p0')
        p0.text = os.path.join(deep_net_initialization_path, 'p0.txt')

        initial_onset_ages = et.SubElement(model_xml, 'initial-onset-ages')
        initial_onset_ages.text = os.path.join(smart_initialization_output_path,
                                               "SmartInitialization_onset_ages.txt")

        initial_log_accelerations = et.SubElement(model_xml, 'initial-log-accelerations')
        initial_log_accelerations.text = os.path.join(smart_initialization_output_path,
                                                      "SmartInitialization_log_accelerations.txt")

        model_xml_path = 'model_after_initialization.xml'
        doc = parseString((et.tostring(model_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

        model_registration_xml = model_xml

        for elt in model_xml.findall('initial-sources'):
            model_xml.remove(elt)

        for elt in model_xml.findall('initial-log-accelerations'):
            model_xml.remove(elt)

        for elt in model_xml.findall('initial-onset-ages'):
            model_xml.remove(elt)

        for elt in model_xml.iter('v0'):
            elt.text = os.path.join('output', 'LongitudinalMetricModel_v0.txt')

        for elt in model_xml.iter('model-type'):
            elt.text = 'LongitudinalMetricRegistration'

        for elt in model_xml.iter('metric-parameters-file'):
            elt.text = os.path.join('output', 'LongitudinalMetricModel_metric_parameters.txt')

        model_xml_path = 'model_registration.xml'
        doc = parseString((et.tostring(model_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')












