import torch

from ...core import default
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata

import logging
logger = logging.getLogger(__name__)


class AffineAtlas(AbstractStatisticalModel):
    """
    Rigid atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dataset, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,
                 gpu_mode=default.gpu_mode,

                 # dataset,
                 freeze_translation_vectors=default.freeze_translation_vectors,
                 freeze_rotation_angles=default.freeze_rotation_angles,
                 freeze_scaling_ratios=default.freeze_scaling_ratios,

                 **kwargs):
        AbstractStatisticalModel.__init__(self, name='AffineAtlas', gpu_mode=gpu_mode)

        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode
        self.number_of_processes = number_of_processes

        # self.dataset = dataset

        object_list, self.objects_name, self.objects_name_extension, self.objects_noise_variance, self.multi_object_attachment \
            = create_template_metadata(template_specifications, self.dimension, gpu_mode=gpu_mode)

        self.template = DeformableMultiObject(object_list)

        self.number_of_subjects = dataset.number_of_subjects
        self.number_of_objects = None

        # Dictionary of booleans.
        self.is_frozen = {'translation_vectors': freeze_translation_vectors, 'rotation_angles': freeze_rotation_angles, 'scaling_ratios': freeze_scaling_ratios}

        # Dictionary of numpy arrays.
        self.fixed_effects['translation_vectors'] = None
        self.fixed_effects['rotation_angles'] = None
        self.fixed_effects['scaling_ratios'] = None

        if dataset is not None and 'landmark_points' in self.template.get_points().keys() and not self.is_frozen['translation_vectors']:
            translations = np.zeros((dataset.number_of_subjects, self.dimension))
            targets = [target[0] for target in dataset.deformable_objects]
            template_mean_point = np.mean(self.template.get_points()['landmark_points'], axis=0)
            for i, target in enumerate(targets):
                target_mean_point = np.mean(target.get_points()['landmark_points'], axis=0)
                translations[i] = target_mean_point - template_mean_point
                self.fixed_effects['translation_vectors'] = translations

        self.update()

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Translations
    def get_translation_vectors(self):
        return self.fixed_effects['translation_vectors']

    def set_translation_vectors(self, t):
        self.fixed_effects['translation_vectors'] = t

    # Rotations
    def get_rotation_angles(self):
        return self.fixed_effects['rotation_angles']

    def set_rotation_angles(self, r):
        self.fixed_effects['rotation_angles'] = r

    # Scalings
    def get_scaling_ratios(self):
        return self.fixed_effects['scaling_ratios']

    def set_scaling_ratios(self, s):
        self.fixed_effects['scaling_ratios'] = s

    # Full fixed effects
    def get_fixed_effects(self):
        out = {}
        if not self.is_frozen['translation_vectors']:
            out['translation_vectors'] = self.fixed_effects['translation_vectors']
        if not self.is_frozen['rotation_angles']:
            out['rotation_angles'] = self.fixed_effects['rotation_angles']
        if not self.is_frozen['scaling_ratios']:
            out['scaling_ratios'] = self.fixed_effects['scaling_ratios']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.is_frozen['translation_vectors']:
            self.set_translation_vectors(fixed_effects['translation_vectors'])
        if not self.is_frozen['rotation_angles']:
            self.set_rotation_angles(fixed_effects['rotation_angles'])
        if not self.is_frozen['scaling_ratios']:
            self.set_scaling_ratios(fixed_effects['scaling_ratios'])

    # For brute force optimization -------------------------------------------------------------------------------------
    # def get_parameters_variability(self):
    #     out = {}
    #     if not self.is_frozen['translation_vectors']:
    #         out['translation_vectors'] = np.zeros(self.fixed_effects['translation_vectors'].shape) + 5.
    #     if not self.is_frozen['rotation_angles']:
    #         out['rotation_angles'] = np.zeros(self.fixed_effects['rotation_angles'].shape) + 0.5
    #     if not self.is_frozen['scaling_ratios']:
    #         out['scaling_ratios'] = np.zeros(self.fixed_effects['scaling_ratios'].shape) + 0.2
    #     return out

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Final initialization steps.
        """

        # self.template.update()
        self.number_of_objects = len(self.template.object_list)

        self._initialize_translation_vectors()
        self._initialize_rotation_angles()
        self._initialize_scaling_ratios()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        translation_vectors = self.get_translation_vectors()
        rotation_angles = self.get_rotation_angles()
        scaling_ratios = self.get_scaling_ratios()

        template_points = {key: self.tensor_scalar_type(value) for key, value in self.template.get_points().items()}
        template_data = {key: self.tensor_scalar_type(value) for key, value in self.template.get_data().items()}

        attachment = 0.0
        targets = [target[0] for target in dataset.deformable_objects]

        if with_grad:
            gradient = {}
            if not self.is_frozen['translation_vectors']:
                gradient['translation_vectors'] = np.zeros(translation_vectors.shape)
            if not self.is_frozen['rotation_angles']:
                gradient['rotation_angles'] = np.zeros(rotation_angles.shape)
            if not self.is_frozen['scaling_ratios']:
                gradient['scaling_ratios'] = np.zeros(scaling_ratios.shape)

            for i, target in enumerate(targets):
                translation_vector_i = self.tensor_scalar_type(translation_vectors[i]).requires_grad_(with_grad and not self.is_frozen['translation_vectors'])
                rotation_angles_i = self.tensor_scalar_type(rotation_angles[i]).requires_grad_(with_grad and not self.is_frozen['rotation_angles'])
                scaling_ratio_i = self.tensor_scalar_type([scaling_ratios[i]]).requires_grad_(with_grad and not self.is_frozen['scaling_ratios'])

                deformed_points = self._deform(translation_vector_i, rotation_angles_i, scaling_ratio_i, template_points)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                attachment_i = self.multi_object_attachment.compute_weighted_distance(deformed_data, self.template, target, self.objects_noise_variance)

                attachment -= attachment_i.detach().cpu().numpy()

                attachment_i.backward()
                if not self.is_frozen['translation_vectors']:
                    gradient['translation_vectors'][i] = - translation_vector_i.grad.detach().cpu().numpy()
                if not self.is_frozen['rotation_angles']:
                    gradient['rotation_angles'][i] = - rotation_angles_i.grad.detach().cpu().numpy()
                if not self.is_frozen['scaling_ratios']:
                    gradient['scaling_ratios'][i] = - scaling_ratio_i.grad.detach().cpu().numpy()

            return attachment, 0.0, gradient

        else:
            for i, target in enumerate(targets):
                translation_vector_i = self.tensor_scalar_type(translation_vectors[i]).requires_grad_(with_grad)
                rotation_angles_i = self.tensor_scalar_type(rotation_angles[i]).requires_grad_(with_grad)
                scaling_ratio_i = self.tensor_scalar_type([scaling_ratios[i]]).requires_grad_(with_grad)

                deformed_points = self._deform(translation_vector_i, rotation_angles_i, scaling_ratio_i, template_points)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                attachment_i = self.multi_object_attachment.compute_weighted_distance(deformed_data, self.template, target, self.objects_noise_variance)

                attachment -= attachment_i.detach().cpu().numpy()

            return attachment, 0.0

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _deform(self, translation_vector, rotation_angles, scaling_ratio, points):
        return self._scale(scaling_ratio, self._rotate(rotation_angles, AffineAtlas._translate(translation_vector, points)))

    def _deform_inverse(self, translation_vector, rotation_angles, scaling_ratio, points):
        return self._scale(1. / scaling_ratio, self._rotate(- rotation_angles, AffineAtlas._translate(- translation_vector, points)))

    @staticmethod
    def _scale(scaling_ratio, points):
        out = {}
        for key, value in points.items():
            if key == 'landmark_points':
                center_of_gravity = torch.mean(value, 0)
                out[key] = scaling_ratio * (value - center_of_gravity) + center_of_gravity
            elif key == 'image_points':
                raise RuntimeError('Not implemented yet.')
            else:
                raise RuntimeError('That\'s unexpected!')
        return out

    def _rotate(self, rotation_angles, points):
        rotation_matrix = self._compute_rotation_matrix(rotation_angles)
        out = {}
        for key, value in points.items():
            if key == 'landmark_points':
                out[key] = torch.matmul(rotation_matrix.unsqueeze(0), value.unsqueeze(self.dimension - 1)).view(-1, self.dimension)
            elif key == 'image_points':
                raise RuntimeError('Not implemented yet.')
            else:
                raise RuntimeError('That\'s unexpected!')
        return out

    @staticmethod
    def _translate(translation_vector, points):
        return {key: value + translation_vector for key, value in points.items()}

    def _compute_rotation_matrix(self, rotation_angles):

        if self.dimension == 2:
            raise RuntimeError('Not implemented yet.')

        elif self.dimension == 3:  # Using Euler angles: https://fr.wikipedia.org/wiki/Matrice_de_rotation
            psi = rotation_angles[0]
            theta = rotation_angles[1]
            phi = rotation_angles[2]

            rot_x = torch.zeros(3, 3).type(self.tensor_scalar_type)
            rot_x[0, 0] = phi.cos()
            rot_x[0, 1] = - phi.sin()
            rot_x[1, 0] = phi.sin()
            rot_x[1, 1] = phi.cos()
            rot_x[2, 2] = 1.

            rot_y = torch.zeros(3, 3).type(self.tensor_scalar_type)
            rot_y[0, 0] = theta.cos()
            rot_y[0, 2] = theta.sin()
            rot_y[2, 0] = - theta.sin()
            rot_y[2, 2] = theta.cos()
            rot_y[1, 1] = 1.

            rot_z = torch.zeros(3, 3).type(self.tensor_scalar_type)
            rot_z[1, 1] = psi.cos()
            rot_z[1, 2] = - psi.sin()
            rot_z[2, 1] = psi.sin()
            rot_z[2, 2] = psi.cos()
            rot_z[0, 0] = 1.

            rotation_matrix = torch.mm(rot_z, torch.mm(rot_y, rot_x))
            return rotation_matrix

    def _initialize_translation_vectors(self):
        assert (self.number_of_subjects > 0)
        if self.get_translation_vectors() is None:
            translation_vectors = np.zeros((self.number_of_subjects, self.dimension))
            self.set_translation_vectors(translation_vectors)
            logger.info('Translation vectors initialized to zero, for ' + str(self.number_of_subjects) + ' subjects.')

    def _initialize_rotation_angles(self):
        assert (self.number_of_subjects > 0)
        if self.get_rotation_angles() is None:
            rotation_angles = np.zeros((self.number_of_subjects, self.dimension))
            self.set_rotation_angles(rotation_angles)
            logger.info('Rotation Euler angles initialized to zero, for ' + str(self.number_of_subjects) + ' subjects.')

    def _initialize_scaling_ratios(self):
        assert (self.number_of_subjects > 0)
        if self.get_scaling_ratios() is None:
            scaling_ratios = np.ones((self.number_of_subjects,))
            self.set_scaling_ratios(scaling_ratios)
            logger.info('Scaling ratios initialized to one, for ' + str(self.number_of_subjects) + ' subjects.')

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir, compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i] for residuals_i in residuals]
            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):

        # Initialize.
        translation_vectors = self.get_translation_vectors()
        rotation_angles = self.get_rotation_angles()
        scaling_ratios = self.get_scaling_ratios()

        template_points = {key: self.tensor_scalar_type(value) for key, value in self.template.get_points().items()}
        template_data = {key: self.tensor_scalar_type(value) for key, value in self.template.get_data().items()}

        targets = [target[0] for target in dataset.deformable_objects]
        residuals = []

        for i, (subject_id, target) in enumerate(zip(dataset.subject_ids, targets)):
            translation_vector_i = self.tensor_scalar_type(translation_vectors[i])
            rotation_angles_i = self.tensor_scalar_type(rotation_angles[i])
            scaling_ratio_i = self.tensor_scalar_type([scaling_ratios[i]])

            # Direct deformation ---------------------------------------------------------------------------------------
            deformed_points = self._deform(translation_vector_i, rotation_angles_i, scaling_ratio_i, template_points)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

            names = []
            for k, (object_name, object_extension) in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            self.template.write(output_dir, names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

            # Inverse deformation --------------------------------------------------------------------------------------
            target_points = {key: self.tensor_scalar_type(value) for key, value in target.get_points().items()}
            target_data = {key: self.tensor_scalar_type(value) for key, value in target.get_data().items()}

            deformed_points = self._deform_inverse(translation_vector_i, rotation_angles_i, scaling_ratio_i, target_points)
            deformed_data = target.get_deformed_data(deformed_points, target_data)

            names = []
            for k, (object_name, object_extension) in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Registration__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            target.write(output_dir, names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, output_dir):
        write_2D_array(self.get_translation_vectors(), output_dir, self.name + "__EstimatedParameters__TranslationVectors.txt")
        write_3D_array(self.get_rotation_angles(), output_dir, self.name + "__EstimatedParameters__RotationAngles.txt")
        write_2D_array(self.get_scaling_ratios(), output_dir, self.name + "__EstimatedParameters__ScalingRatios.txt")
