import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import math
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from numpy.random import poisson, exponential, normal
import matplotlib.pyplot as plt
import warnings

from deformetrica import get_model_options
from api.deformetrica import Deformetrica
from core.models.longitudinal_atlas import LongitudinalAtlas
from core.models.clustered_longitudinal_atlas import ClusteredLongitudinalAtlas
from in_out.dataset_functions import create_template_metadata


from in_out.xml_parameters import XmlParameters
from core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from launch.estimate_longitudinal_metric_model import instantiate_longitudinal_metric_model
from in_out.deformable_object_reader import DeformableObjectReader
from in_out.dataset_functions import create_dataset
from in_out.array_readers_and_writers import *


def add_gaussian_noise_to_vtk_file(global_output_dir, filename, obj_type, noise_std):
    reader = DeformableObjectReader()
    obj = reader.create_object(filename, obj_type)
    obj.update()
    obj.set_points(obj.points + normal(0.0, noise_std, size=obj.points.shape))
    obj.write(global_output_dir, os.path.basename(filename))


def main(arg, model_xml_path, number_of_subjects, mean_number_of_visits_minus_two, mean_observation_time_window, global_add_noise, classes):

    """
    Basic info printing.
    """

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')
    print('')

    """
    Read command line, create output directory, read the model xml file.
    """

    sample_index = 1
    sample_folder = 'sample_' + str(sample_index)
    while os.path.isdir(sample_folder):
        sample_index += 1
        sample_folder = '/Users/local_vianneydebavelaere/Documents/Thèse/Python/Results/starmen_for_simu/test/sample' + str(sample_index)
    os.mkdir(sample_folder)
    global_output_dir = sample_folder

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)

    template_specifications = xml_parameters.template_specifications
    nb_classes = np.max(classes) + 1
    model_options = get_model_options(xml_parameters)
    model_options['tensor_scalar_type'] = torch.DoubleTensor
    model_options['tensor_integer_type'] = torch.LongTensor

    global_dimension = model_options['dimension']

    # deformetrica = Deformetrica()
    # (template_specifications, model_options, _) = deformetrica.further_initialization(
    #     xml_parameters.model_type, xml_parameters.template_specifications, get_model_options(xml_parameters))

    if xml_parameters.model_type == 'ClusteredLongitudinalAtlas'.lower():

        """
        Instantiate the model.
        """
        # if np.min(model.get_noise_variance()) < 0:
        #     model.set_noise_variance(np.array([0.0]))

        #path = '/Users/local_vianneydebavelaere/Documents/Thèse/Python/deformetrica-last/sandbox/longitudinal_atlas/landmark/3d/hippocampi'
        #xml_parameters.read_all_xmls(path + '/model.xml', path + '/data_set_93subjects.xml',
        #                             path + '/optimization_parameters.xml', path + '/output/')
        #nb_subjects = 93

        """
        Draw random visit ages and create a degenerated dataset object.
        """

        onset_ages = np.zeros(number_of_subjects)
        accelerations = np.zeros([3,number_of_subjects])

        visit_ages = []
        for i in range(number_of_subjects):
            number_of_visits = 2 + poisson(mean_number_of_visits_minus_two)
            # observation_time_window = exponential(mean_observation_time_window)/10
            observation_time_window = 5

            time_between_two_consecutive_visits_before = observation_time_window / float(number_of_visits)
            time_between_two_consecutive_visits_after = observation_time_window / float(number_of_visits)

            #age_at_baseline = normal(68, math.sqrt(model.get_time_parameters_variance(0)[2,2])) \
            #                  - 0.5 * observation_time_window

            if classes[i] == 0: age_at_baseline = 70
            else: age_at_baseline = 70


            ages = [age_at_baseline - (j+1) * time_between_two_consecutive_visits_before for j in range(number_of_visits)]
            for j in range(number_of_visits):
                ages.append(age_at_baseline + (j+1) * time_between_two_consecutive_visits_after)
            nb_visits = np.random.randint(5,10,1)
            ages = np.sort(ages)
            ages = np.linspace(60,83,nb_visits)
            visit_ages.append(ages)

        #visit_ages = [np.linspace(60,80,20)]

        mini = min(visit_ages[0])
        maxi = max(visit_ages[0])
        for k in range(1, visit_ages.__len__()):
            if min(visit_ages[k]) < mini: mini = min(visit_ages[k])
            if max(visit_ages[k]) > maxi: maxi = max(visit_ages[k])

        model = ClusteredLongitudinalAtlas(template_specifications, min_times=mini, max_times=maxi, **model_options)

        subject_ids = ['s' + str(i) for i in range(number_of_subjects)]
        dataset = LongitudinalDataset(subject_ids, times=visit_ages)
        # dataset = LongitudinalDataset(['0'], times=[[0,1]])

        print('>> %d subjects will be generated, with %.2f visits on average, covering an average period of %.2f years.'
              % (number_of_subjects, float(dataset.total_number_of_observations) / float(number_of_subjects),
                 np.mean(np.array([ages[-1] - ages[0] for ages in dataset.times]))))

        """
        Generate individual RER.
        """

        # Complementary xml parameters.
        # tmin = xml_parameters.tmin
        # tmax = xml_parameters.tmax
        tmin = 0
        tmax = 1

        if tmin == float('inf'):
            tmin *= -1
        if tmax == - float('inf'):
            tmax *= -1

        sources_mean = 0.0
        sources_std = 0.2
        if xml_parameters.initial_sources_mean is not None:
            sources_mean = read_2D_array(xml_parameters.initial_sources_mean)
        if xml_parameters.initial_sources_std is not None:
            sources_std = read_2D_array(xml_parameters.initial_sources_std)

        sources = np.zeros((number_of_subjects, model.number_of_sources)) + sources_mean

        #dir = '/Users/local_vianneydebavelaere/Documents/Thèse/Python/Results/output_93subjects/'
        #classes = open(dir + 'LongitudinalAtlas__EstimatedParameters__Classes.txt').read().replace('\n', ' ').split(
         #   ' ')[:-1]


        # min_age = np.zeros(number_of_subjects)
        # max_age = np.zeros(number_of_subjects)

        A = 0.05*np.eye(3)
        A[-1,-1] = 1
        model.individual_random_effects['time_parameters'][0].set_covariance(A)
        # model.individual_random_effects['time_parameters'][1].set_covariance(A)
        model.individual_random_effects['time_parameters'][0].set_mean(np.array([0,0,70]))
        # model.individual_random_effects['time_parameters'][1].set_mean(np.array([0,0,70]))

        #onset_ages = open(dir + 'LongitudinalAtlas__EstimatedParameters__OnsetAges.txt').read().replace('\n', ',').split(',')[:-1]
        #rupture_time = []
        #rupture_time.append(float(open(dir + 'LongitudinalAtlas__EstimatedParameters__RuptureTime_classe0.txt').read()))
        #rupture_time.append(float(open(dir + 'LongitudinalAtlas__EstimatedParameters__RuptureTime_classe1.txt').read()))
        #accelerations = open(dir + 'LongitudinalAtlas__EstimatedParameters__Accelerations.txt').read().replace('\n',
         #                                                                                                      ' ').split(
         #   ' ')[:-1]
        #accelerations2 = open(dir + 'LongitudinalAtlas__EstimatedParameters__Accelerations2.txt').read().replace('\n',
        #                                                                                                         ' ').split(
        #    ' ')[:-1]
        #sources = open(dir + 'LongitudinalAtlas__EstimatedParameters__Sources.txt').read().replace('\n',
        #                                                                                                ',').split(',')[
        #             :-1]

        i = 0
        sources_std = 1
        while i in range(number_of_subjects):
            # [accelerations[0, i], accelerations[1, i], onset_ages[i]] = model.individual_random_effects['time_parameters'][0].sample()
            [accelerations[0, i], accelerations[1, i], accelerations[2,i], onset_ages[i]] = [0,0,0,0]
            sources[i] = model.individual_random_effects['sources'].sample() * sources_std/4
            i += 1

            # min_age[i] = tR[classes[i]] - np.exp(accelerations[i]) * (-visit_ages[i][0] + tR[classes[i]] + onset_ages[i])
            # max_age[i] = tR[classes[i]] + np.exp(accelerations2[i]) * (visit_ages[i][-1] - tR[classes[i]] - onset_ages[i])
            # if visit_ages[i][0] < onset_ages[i] and visit_ages[i][-1] > onset_ages[i]:
            #     i += 1

        dataset.times = visit_ages
        model.name = 'SimulatedData'



        model.set_rupture_time(65,0)
        model.set_rupture_time(75,1)

        # sources = np.zeros([1, 32])
        model.set_modulation_matrix(np.ones([12,4]), 0)
        # model.set_modulation_matrix(np.ones([12,4]), 1)
        # sources[0,2] = -0.5
        # sources[0,14] = -0.5

        individual_RER = {}
        # individual_RER['sources'] = sources
        # individual_RER['onset_ages'] = np.array([70.])
        # individual_RER['accelerations'] = np.array([1.])
        # individual_RER['accelerations2'] = np.array([1.])
        # individual_RER['classes'] = np.array([0])

        individual_RER['sources'] = sources
        individual_RER['onset_ages'] = onset_ages
        individual_RER['accelerations'] = np.transpose(accelerations)
        individual_RER['classes'] = classes

        """
        Call the write method of the model.
        """

        momenta = []
        mom = [None]*model.nb_classes
        # for l in range(model.nb_tot_component):
            # momenta.append(np.random.normal(0,0.01,model.get_control_points(0).size).reshape(np.shape(model.get_control_points(0))))
            # momenta.append([0,1]*model.get_control_points(0).size/2)
        momenta.append([[0,1],[0,0], [0,0], [0,0], [0,0], [0,0]])
        momenta.append([[0,0],[0,1], [0,0], [0,0], [0,0], [0,0]])
        momenta.append([[0,0],[0,-1], [0,0], [0,0], [0,1], [0,0]])

        for k in range(model.nb_classes):
            mom[k] = []
            for l in model.num_component[k]:
                mom[k].append(momenta[l])
            model.set_momenta(0.1*np.array(mom[k]),k)

        cp = model.get_control_points(0)
        cp[4,:] = [1.5,0.7]
        model.set_control_points(cp,0)

        cp = model.get_control_points(0)
        width_x = np.abs(cp[0, 0] - cp[-1, 0]) / 8
        width_y = np.abs(cp[0, 1] - cp[-1, 1]) / 8
        x, y = np.meshgrid(np.linspace(cp[0, 0] - 3 * width_x, cp[-1, 0] + 3 * width_x, 15),
                           np.linspace(cp[0, 1] - 3 * width_y, cp[-1, 1] + 3 * width_y, 15))
        test = np.array([x, y])
        intensity = np.zeros([2, 15, 15])
        momenta = model.get_momenta(0, 0)

        for i in range(15):
            for j in range(15):
                for k in range(int(cp.size / 2)):
                    intensity[:, i, j] += np.exp(-np.linalg.norm(test[:, i, j] - cp[k, :]) / 0.8 ** 2) * np.array(
                        momenta[k])

        plt.quiver(x, y, intensity[0, :, :], intensity[1, :, :], scale=17, color='blue')
        cpx = cp[:, 0]
        cpy = cp[:, 1]
        momx = np.array(momenta)[:, 0]
        momy = np.array(momenta)[:, 1]
        plt.quiver(cpx, cpy, momx, momy, scale=17, color='red')
        plt.scatter(cp[:, 0], cp[:, 1], color='red')
        template = model.get_template_data(0)['landmark_points']
        template_closed = np.zeros([template.shape[0] + 1, 2])
        template_closed[:-1, :] = template
        template_closed[-1, :] = template[0, :]
        plt.plot(template_closed[:, 0], template_closed[:, 1], color='black')
        plt.axis([-2, 2.5, -2, 2.5])
        plt.show()

        model.write(dataset, None, individual_RER, global_output_dir, update_fixed_effects=False, write_residuals=False)

        # template_fin = model.spatiotemporal_reference_frame[0].get_template_points_exponential_parameters(1,np.zeros(sources.shape))

        if global_dimension == 2:
            cmd_replace = 'sed -i -- s/POLYGONS/LINES/g ' + global_output_dir + '/*Reconstruction*'
            cmd_delete = 'rm ' + global_output_dir + '/*--'
            cmd = cmd_replace + ' && ' + cmd_delete
            os.system(cmd)  # Quite time-consuming.

        """
        Optionally add gaussian noise to the generated samples.
        """

        #model.set_noise_variance(np.array(10))

        if global_add_noise:
            assert np.min(model.get_noise_variance()) > 0, 'Invalid noise variance.'
            objects_type = [elt['deformable_object_type'] for elt in xml_parameters.template_specifications.values()]
            for i in range(number_of_subjects):
                for j, age in enumerate(dataset.times[i]):
                    for k, (obj_type, obj_name, obj_extension, obj_noise) in enumerate(zip(
                            objects_type, model.objects_name, model.objects_name_extension,
                            model.get_noise_variance())):
                        filename = sample_folder + '/SimulatedData__Reconstruction__%s__subject_s%d__tp_%d__age_%.2f%s' \
                                   % (obj_name, i, j, age, obj_extension)
                        add_gaussian_noise_to_vtk_file(global_output_dir, filename, obj_type, math.sqrt(obj_noise))

            if global_dimension == 2:
                cmd_replace = 'sed -i -- s/POLYGONS/LINES/g ' + global_output_dir + '/*Reconstruction*'
                cmd_delete = 'rm ' + global_output_dir + '/*--'
                cmd = cmd_replace + ' && ' + cmd_delete
                os.system(cmd)  # Quite time-consuming.

        """
        Create and save the dataset xml file.
        """

        dataset_xml = et.Element('data-set')
        dataset_xml.set('deformetrica-min-version', "3.0.0")

        for i in range(number_of_subjects):

            subject_id = 'sub-' + str(i)
            subject_xml = et.SubElement(dataset_xml, 'subject')
            subject_xml.set('id', subject_id)

            for j, age in enumerate(dataset.times[i]):

                visit_id = 'ses-' + str(j)
                visit_xml = et.SubElement(subject_xml, 'visit')
                visit_xml.set('id', visit_id)
                age_xml = et.SubElement(visit_xml, 'age')
                age_xml.text = '%.2f' % age

                for k, (obj_name, obj_extension) in enumerate(zip(model.objects_name, model.objects_name_extension)):
                    filename_xml = et.SubElement(visit_xml, 'filename')
                    filename_xml.text = sample_folder + '/SimulatedData__Reconstruction__%s__subject_s%d__tp_%d__age_%.2f%s' \
                                        % (obj_name, i, j, age, obj_extension)
                    filename_xml.set('object_id', obj_name)

        dataset_xml_path = sample_folder + 'data_set__sample_' + str(sample_index) + '.xml'
        doc = parseString((et.tostring(dataset_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(dataset_xml_path, [doc], fmt='%s')

        """
        Create a dataset object from the xml, and compute the residuals.
        """

        xml_parameters._read_dataset_xml(dataset_xml_path)
        dataset = create_dataset(xml_parameters.template_specifications,
                                 visit_ages=xml_parameters.visit_ages,
                                 dataset_filenames=xml_parameters.dataset_filenames,
                                 subject_ids=xml_parameters.subject_ids,
                                 dimension=global_dimension)

        # if global_add_noise:
        #     control_points, backward_momenta, forward_momenta, modulation_matrix, rupture_time = model._fixed_effects_to_torch_tensors(False)
        #     sources, onset_ages, accelerations, accelerations2 = model._individual_RER_to_torch_tensors(individual_RER, False)
        #     template_points, template_data = model._template_to_torch_tensors(False)
        #     absolute_times, tmin, tmax = model._compute_absolute_times(dataset.times, onset_ages, accelerations, accelerations2, rupture_time)
        #     model._update_spatiotemporal_reference_frame(
        #         template_points, control_points, backward_momenta, forward_momenta, modulation_matrix,
        #         tmin, tmax)
        #     residuals = model._compute_residuals(dataset, template_data, absolute_times, sources)
        #
        #     residuals_list = [[[residuals_i_j_k.detach().cpu().numpy() for residuals_i_j_k in residuals_i_j]
        #                        for residuals_i_j in residuals_i] for residuals_i in residuals]
        #     write_3D_list(residuals_list, global_output_dir, model.name + "__EstimatedParameters__Residuals.txt")
        #
        #     # Print empirical noise if relevant.
        #     assert np.min(model.get_noise_variance()) > 0, 'Invalid noise variance.'
        #     objects_empirical_noise_std = np.zeros((len(residuals_list[0][0])))
        #     for i in range(len(residuals_list)):
        #         for j in range(len(residuals_list[i])):
        #             for k in range(len(residuals_list[i][j])):
        #                 objects_empirical_noise_std[k] += residuals_list[i][j][k]
        #     for k in range(len(residuals_list[0][0])):
        #         objects_empirical_noise_std[k] = \
        #             math.sqrt(objects_empirical_noise_std[k]
        #                       / float(dataset.total_number_of_observations * model.objects_noise_dimension[k]))
        #         print('>> Empirical noise std for object "%s": %.4f'
        #               % (model.objects_name[k], objects_empirical_noise_std[k]))
        #     write_2D_array(objects_empirical_noise_std,
        #                    global_output_dir, model.name + '__EstimatedParameters__EmpiricalNoiseStd.txt')

    elif xml_parameters.model_type == 'LongitudinalMetricLearning'.lower():

        """
        Instantiate the model.
        """
        model, _ = instantiate_longitudinal_metric_model(xml_parameters, dataset=None,
                                                         number_of_subjects=number_of_subjects,
                                                         observation_type='image')
        assert model.get_noise_variance() is not None \
               and model.get_noise_variance() > 0., "Please provide a noise variance"

        """
        Draw random visit ages and create a degenerated dataset object.
        """

        visit_ages = []
        for i in range(number_of_subjects):
            number_of_visits = 2 + poisson(mean_number_of_visits_minus_two)
            observation_time_window = exponential(mean_observation_time_window)

            time_between_two_consecutive_visits = observation_time_window / float(number_of_visits - 1)
            age_at_baseline = normal(model.get_reference_time(), math.sqrt(model.get_onset_age_variance())) \
                              - 0.5 * observation_time_window

            ages = [age_at_baseline + j * time_between_two_consecutive_visits for j in range(number_of_visits)]
            visit_ages.append(np.array(ages))

        dataset = LongitudinalDataset()
        dataset.times = visit_ages
        dataset.subject_ids = ['s' + str(i) for i in range(number_of_subjects)]
        dataset.number_of_subjects = number_of_subjects
        dataset.total_number_of_observations = sum([len(elt) for elt in visit_ages])

        print('>> %d subjects will be generated, with %.2f visits on average, covering an average period of %.2f years.'
              % (number_of_subjects, float(dataset.total_number_of_observations) / float(number_of_subjects),
                 np.mean(np.array([ages[-1] - ages[0] for ages in dataset.times]))))

        """
        Generate metric parameters.
        """
        if xml_parameters.metric_parameters_file is None:
            print("The generation of metric parameters is only handled in one dimension")
            values = np.random.binomial(1, 0.5, xml_parameters.number_of_interpolation_points)
            values = values / np.sum(values)
            model.set_metric_parameters(values)

        """
        Generate individual RER.
        """

        onset_ages = np.zeros((number_of_subjects,))
        log_accelerations = np.zeros((number_of_subjects,))
        sources = np.zeros((number_of_subjects, xml_parameters.number_of_sources))

        for i in range(number_of_subjects):
            onset_ages[i] = model.individual_random_effects['onset_age'].sample()
            log_accelerations[i] = model.individual_random_effects['log_acceleration'].sample()
            sources[i] = model.individual_random_effects['sources'].sample()

        individual_RER = {}
        individual_RER['onset_age'] = onset_ages
        individual_RER['log_acceleration'] = log_accelerations
        individual_RER['sources'] = sources

        """
        Call the write method of the model.
        """

        model.name = 'SimulatedData'
        model.write(dataset, None, individual_RER, sample=True)

        # Create a dataset xml for the simulations which will
        dataset_xml = et.Element('data-set')
        dataset_xml.set('deformetrica-min-version', "3.0.0")

        if False:
            group_file = et.SubElement(dataset_xml, 'group-file')
            group_file.text = "sample_%d/SimulatedData_subject_ids.txt" % (sample_index)

            observations_file = et.SubElement(dataset_xml, 'observations-file')
            observations_file.text = "sample_%d/SimulatedData_generated_values.txt" % (sample_index)

            timepoints_file = et.SubElement(dataset_xml, 'timepoints-file')
            timepoints_file.text = "sample_%d/SimulatedData_times.txt" % (sample_index)

            dataset_xml_path = 'data_set__sample_' + str(sample_index) + '.xml'
            doc = parseString(
                (et.tostring(dataset_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
            np.savetxt(dataset_xml_path, [doc], fmt='%s')

        else:  # Image dataset
            dataset_xml = et.Element('data-set')
            dataset_xml.set('deformetrica-min-version', "3.0.0")

            for i in range(number_of_subjects):

                subject_id = 's' + str(i)
                subject_xml = et.SubElement(dataset_xml, 'subject')
                subject_xml.set('id', subject_id)

                for j, age in enumerate(dataset.times[i]):
                    visit_id = 'ses-' + str(j)
                    visit_xml = et.SubElement(subject_xml, 'visit')
                    visit_xml.set('id', visit_id)
                    age_xml = et.SubElement(visit_xml, 'age')
                    age_xml.text = str(age)

                    filename_xml = et.SubElement(visit_xml, 'filename')
                    filename_xml.set('object_id', 'starfish')
                    filename_xml.text = os.path.join(global_output_dir, 'subject_'+str(i),
                                                 model.name + "_" + str(dataset.subject_ids[i])+ "_t__" + str(age) + ".npy")


            dataset_xml_path = 'data_set__sample_' + str(sample_index) + '.xml'
            doc = parseString(
                (et.tostring(dataset_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
            np.savetxt(dataset_xml_path, [doc], fmt='%s')

    else:
        msg = 'Sampling from the specified "' + xml_parameters.model_type + '" model is not available yet.'
        raise RuntimeError(msg)


real_c = [0,1]*50
for i in [97,94,79,70,64,19,18]:
    real_c.pop(i)
main(0, '/Users/local_vianneydebavelaere/Documents/Thèse/Python/Results/starmen_for_simu/model_forsample.xml', 1, 1, 10, True, np.array([0]))
