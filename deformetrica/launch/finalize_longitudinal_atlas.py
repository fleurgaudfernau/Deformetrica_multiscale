import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')


import fnmatch
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString

from ..in_out.xml_parameters import XmlParameters
from ..in_out.dataset_functions import create_template_metadata
from ..in_out.array_readers_and_writers import *


def insert_model_xml_level1_entry(model_xml_level0, key, value):
    found_tag = False
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == key:
            model_xml_level1.text = value
            found_tag = True
    if not found_tag:
        new_element_xml = et.SubElement(model_xml_level0, key)
        new_element_xml.text = value
    return model_xml_level0


def insert_model_xml_template_spec_entry(model_xml_level0, key, values):
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'template':
            k = -1
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == 'object':
                    k += 1
                    found_tag = False
                    for model_xml_level3 in model_xml_level2:
                        if model_xml_level3.tag.lower() == key.lower():
                            model_xml_level3.text = values[k]
                            found_tag = True
                    if not found_tag:
                        new_element_xml = et.SubElement(model_xml_level2, key)
                        new_element_xml.text = values[k]
    return model_xml_level0


def insert_model_xml_deformation_parameters_entry(model_xml_level0, key, value):
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'deformation-parameters':
            found_tag = False
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == key:
                    model_xml_level2.text = value
                    found_tag = True
            if not found_tag:
                new_element_xml = et.SubElement(model_xml_level1, key)
                new_element_xml.text = value
    return model_xml_level0


def finalize_longitudinal_atlas(model_xml_path, output_dir='output'):

    """
    0]. Read command line, read original xml parameters.
    """

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)

    """
    1]. Create the finalized_model.xml file.
    """

    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0, 'model-type', 'LongitudinalRegistration')

    longitudinal_atlas_output_path = os.path.relpath(output_dir, os.getcwd())

    global_objects_name, global_objects_name_extension \
        = create_template_metadata(xml_parameters.template_specifications)[1:3]

    # Template.
    estimated_template_objects_path = []
    for k, (object_name, object_name_extension) in enumerate(zip(global_objects_name,
                                                                 global_objects_name_extension)):
        estimated_template_path = os.path.join(longitudinal_atlas_output_path, fnmatch.filter(
            os.listdir(longitudinal_atlas_output_path),
            'LongitudinalAtlas__EstimatedParameters__Template_' + object_name + '*' + object_name_extension)[0])
        estimated_template_objects_path.append(estimated_template_path)

    model_xml_level0 = insert_model_xml_template_spec_entry(
        model_xml_level0, 'filename', estimated_template_objects_path)

    # Control points.
    estimated_control_points_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ControlPoints.txt')
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-control-points', estimated_control_points_path)

    # Momenta.
    estimated_momenta_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__Momenta.txt')
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-momenta', estimated_momenta_path)

    # Modulation matrix.
    estimated_modulation_matrix_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ModulationMatrix.txt')
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-modulation-matrix', estimated_modulation_matrix_path)

    # Reference time.
    estimated_reference_time_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ReferenceTime.txt')
    estimated_reference_time = np.loadtxt(estimated_reference_time_path)
    model_xml_level0 = insert_model_xml_deformation_parameters_entry(
        model_xml_level0, 't0', '%.4f' % estimated_reference_time)

    # Time-shift variance.
    estimated_time_shift_std_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__TimeShiftStd.txt')
    estimated_time_shift_std = np.loadtxt(estimated_time_shift_std_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-time-shift-std', '%.4f' % estimated_time_shift_std)

    # Acceleration variance.
    estimated_acceleration_std_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__AccelerationStd.txt')
    estimated_acceleration_std = np.loadtxt(estimated_acceleration_std_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-acceleration-std', '%.4f' % estimated_acceleration_std)

    # Noise variance.
    estimated_noise_std_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__NoiseStd.txt')
    if len(global_objects_name) == 1:
        estimated_noise_std = [np.loadtxt(estimated_noise_std_path)[()]]
    else:
        estimated_noise_std = np.loadtxt(estimated_noise_std_path)
    global_initial_noise_std_string = ['{:.4f}'.format(elt) for elt in estimated_noise_std]
    model_xml_level0 = insert_model_xml_template_spec_entry(
        model_xml_level0, 'noise-std', global_initial_noise_std_string)

    # Finalization.
    model_xml_path = 'finalized_model.xml'
    doc = parseString(
        (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')
