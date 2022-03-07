import math
import time
import copy
import torch
import numpy as np
import pywt
import itertools
from scipy.ndimage import gaussian_filter
from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
from ...support import utilities

from ...support.utilities.wavelets import WaveletTransform, haar_forward, haar_IWT_mat, haar_FWT_mat, haar_inverse_transpose

logger = logging.getLogger(__name__)

def _silence_fine_or_smooth_zones(current_scale, zones, silent_coarse_momenta, freeze_control_points, gradient_of_coef, haar_dec_of_points):
    
    indices_to_browse_along_dim = [list(range(e)) for e in list(gradient_of_coef.wc.shape)]

    for indices in itertools.product(*indices_to_browse_along_dim):
        position, type, scale = gradient_of_coef.haar_coeff_pos_type([i for i in indices])

        #silence finer zones that we haven't reached yet
        if scale < current_scale: #the higher the scale, the coarser
            gradient_of_coef.wc[indices] = 0 #fine scales : only ad, da, dd
            
            if not freeze_control_points:
                haar_dec_of_points.wc[indices] = 0

        #silence smooth zone
        elif silent_coarse_momenta and scale in silent_coarse_momenta.keys():
            zones_to_silence_at_scale = silent_coarse_momenta[scale]
            positions_to_silence_at_scale = [zones[scale][k]["position"] for k in zones_to_silence_at_scale]
            if position in positions_to_silence_at_scale and type != ['L', 'L']:
                gradient_of_coef.wc[indices] = 0
                
                if not freeze_control_points:
                    haar_dec_of_points.wc[indices] = 0
    
    return gradient_of_coef, haar_dec_of_points    

def _compute_haar_transform_of_gradients(dimension, gradient, zones, current_scale, silent_coarse_momenta, points_per_axis, freeze_control_points):
    gradient['coarse_momenta'] = []

    if not freeze_control_points:
        gradient['coarse_points'] = []   

    for s in range(gradient['momenta'].shape[0]): #for each subject
        gradient_of_sub = []
        gradient_of_points = []
        for d in range(dimension):
            gradient_of_momenta = gradient['momenta'][s, :, d].reshape(tuple(points_per_axis))
            #gradient_of_coef = haar_forward(gradient_of_momenta) #before 13/12
            gradient_of_coef = haar_inverse_transpose(gradient_of_momenta, J = None, gamma = 1)
            #gradient_of_coef = haar_forward(gradient_of_momenta, J = None, gamma = 0) 

            if not freeze_control_points:
                gradient_of_cp = gradient['control_points'][:, d].reshape(tuple(points_per_axis))
                haar_dec_of_points = haar_forward(gradient_of_cp)
            else:
                haar_dec_of_points = []

            gradient_of_coef, haar_dec_of_points = _silence_fine_or_smooth_zones(current_scale, zones, silent_coarse_momenta, freeze_control_points, gradient_of_coef, haar_dec_of_points)   
            
            gradient_of_sub.append(gradient_of_coef)
            if not freeze_control_points:
                gradient_of_points.append(haar_dec_of_points)
        
        gradient['coarse_momenta'].append(gradient_of_sub) #list of n lists of 3 list of haar wavelets objects
        
        if not freeze_control_points:
            gradient['coarse_points'] = gradient_of_points
    
    return gradient

def _subject_attachment_and_regularity(arg):
    """
    Auxiliary function for multithreading (cannot be a class method).
    """
    from .abstract_statistical_model import process_initial_data
    if process_initial_data is None:
        raise RuntimeError('process_initial_data is not set !')

    # Read arguments.
    (deformable_objects, multi_object_attachment, objects_noise_variance,
     freeze_template, freeze_control_points, freeze_momenta,
     exponential, sobolev_kernel, use_sobolev_gradient, tensor_scalar_type, gpu_mode) = process_initial_data
    (i, template, template_data, control_points, momenta, with_grad) = arg

    # start = time.perf_counter()
    device, device_id = utilities.get_best_device(gpu_mode=gpu_mode)
    # device, device_id = ('cpu', -1)
    if device_id >= 0:
        torch.cuda.set_device(device_id)

    # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
    template_data = {key: utilities.move_data(value, device=device, dtype=tensor_scalar_type,
                                              requires_grad=with_grad and not freeze_template)
                     for key, value in template_data.items()}
    template_points = {key: utilities.move_data(value, device=device, dtype=tensor_scalar_type,
                                                requires_grad=with_grad and not freeze_template)
                       for key, value in template.get_points().items()}
    control_points = utilities.move_data(control_points, device=device, dtype=tensor_scalar_type,
                                         requires_grad=with_grad and not freeze_control_points)
    momenta = utilities.move_data(momenta, device=device, dtype=tensor_scalar_type,
                                  requires_grad=with_grad and not freeze_momenta)

    assert torch.device(
        device) == control_points.device == momenta.device, 'control_points and momenta tensors must be on the same device. ' \
                                                            'device=' + device + \
                                                            ', control_points.device=' + str(control_points.device) + \
                                                            ', momenta.device=' + str(momenta.device)

    attachment, regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
        exponential, template_points, control_points, momenta,
        template, template_data, multi_object_attachment,
        deformable_objects[i], objects_noise_variance,
        device) #ajout fg

    res = DeterministicAtlas._compute_gradients(
        attachment, regularity, template_data,
        freeze_template, template_points,
        freeze_control_points, control_points,
        freeze_momenta, momenta,
        optimize_nb_control_points, coarse_momenta, silent_coarse_momenta, zones, simension, points_per_axis, current_scale,
        use_sobolev_gradient, sobolev_kernel,
        freeze_control_points,
        with_grad)

    # elapsed = time.perf_counter() - start
    # logger.info('pid=' + str(os.getpid()) + ', ' + torch.multiprocessing.current_process().name +
    #       ', device=' + device + ', elapsed=' + str(elapsed))
    
    return i, res 

class DeterministicAtlas(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, number_of_subjects,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 optimize_nb_control_points = default.optimize_nb_control_points, #ajout fg
                 max_spacing = default.max_spacing, #ajout fg
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 gpu_mode=default.gpu_mode,
                 process_per_gpu=default.process_per_gpu,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='DeterministicAtlas', number_of_processes=number_of_processes,
                                          gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points
        self.optimize_nb_control_points = optimize_nb_control_points #ajout fg
        self.max_spacing = max_spacing #ajout fg
        self.freeze_momenta = freeze_momenta

        #ajout fg
        self.initial_cp_spacing = initial_cp_spacing #determines nb of points 
        self.gpu_mode = gpu_mode
        self.deformation_kernel_width = deformation_kernel_width
        if self.optimize_nb_control_points:
            self.deformation_kernel_width = self.initial_cp_spacing
        self.deformation_kernel_type = deformation_kernel_type
        self.initial_residuals = 0

        self.fixed_effects['coarse_momenta'] = None
        self.haar_matrix = None
        self.silent_coarse_momenta = dict()
        self.silent_coarse_momenta[0] = []
        self.points_per_axis = None
        self.current_scale = 0


        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications,
                                                                                               self.dimension)

        self.template = DeformableMultiObject(object_list)
        # self.template.update()
        self.number_of_objects = len(self.template.object_list)
        
        # Deformation.
        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(deformation_kernel_type,
                                          gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)


        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type,
                                                         gpu_mode=gpu_mode,
                                                         kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, self.initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
            initial_momenta, self.number_of_control_points, self.dimension, number_of_subjects)
        self.number_of_subjects = number_of_subjects

        self.process_per_gpu = process_per_gpu

        #ajout fg
        self.deformation_kernel_width = deformation_kernel_width
        self.points_per_axis = [len(set(list(self.fixed_effects['control_points'][:, k]))) for k in range(self.dimension)]
        if self.dimension == 3:
            self.points_per_axis = [self.points_per_axis[1], self.points_per_axis[0], self.points_per_axis[2]]
        self.zones = dict()
        
        self.iterations_coarse_to_fine = []
        self.fixed_effects['coarse_momenta'] = [] #list of n lists of 3 haar 
        self.shape = self.fixed_effects['template_data']['image_intensities'].shape
        
        print("self.points_per_axis", self.points_per_axis, "shape", self.shape)

        if self.optimize_nb_control_points:
            
            self.initialize_coarse_to_fine_momenta()

            if not self.freeze_control_points:
                self.initialize_coarse_to_fine_moving_points()
            
            #self.initialize_coarse_to_fine_template()

    def initialize_coarse_to_fine_momenta(self):
        print("\n Initialisation - coarse to fine on momenta")

        for s in range(self.fixed_effects['momenta'].shape[0]):
            coefficients_sub = [haar_forward(self.fixed_effects['momenta'][s, :, d].reshape(tuple(self.points_per_axis)),
                                            gamma = 1) for d in range(self.dimension)]
            self.fixed_effects['coarse_momenta'].append(coefficients_sub)

        self.current_scale = self.fixed_effects['coarse_momenta'][0][0].J
        self.coarser_scale = self.fixed_effects['coarse_momenta'][0][0].J
        self.zones[self.current_scale] = dict()
        self.silent_coarse_momenta[self.current_scale] = []

    def initialize_coarse_to_fine_moving_points(self):
        print("\n Initialisation - coarse to fine on moving control points")

        self.fixed_effects['coarse_points'] = [haar_forward(self.fixed_effects['control_points'][:, d].reshape(tuple(self.points_per_axis))) \
                                                for d in range(self.dimension)]
        print("self.fixed_effects['control_points']", self.fixed_effects['control_points'])
        print("self.fixed_effects['coarse_points']", self.fixed_effects['coarse_points'])
    

    def initialize_coarse_to_fine_template(self):

        print("\n Initialisation - coarse to fine on template")
        
        self.fixed_effects['original_template_data'] = self.fixed_effects['template_data'].copy() #np array
        
        #### gaussian filter

        #self.current_image_scale = int(self.shape[0]/20)
        self.current_image_scale = 2
        new_intensities = gaussian_filter(self.fixed_effects['template_data']["image_intensities"], sigma = self.current_image_scale)
        

        ###haar filter
        """
        difference_scales_points_voxels = self.difference_scales_points_voxels()
        #if difference_scales_points_voxels > 0:
        #    self.current_image_scale = self.current_scale + difference_scales_points_voxels - 1
        #else:
        #self.current_image_scale = self.current_scale #same scale as points...
        #we want more or less to start at original resolution / 10
        #size = [2**scale for d in range self.dimension]
        self.current_image_scale = 3
        
        print("self.current_image_scale", self.current_image_scale)
        intensities = self.fixed_effects['template_data']["image_intensities"]
        new_intensities = self.haar_transform_and_filter_intensities(intensities)"""
        
        self.fixed_effects['template_data']["image_intensities"] = new_intensities
        self.template.set_data(self.fixed_effects['template_data'])


    def initialize_noise_variance(self, dataset, device='cpu'):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, control_points, momenta, coarse_momenta = self._fixed_effects_to_torch_tensors(False,
                                                                                                           device=device)
            targets = dataset.deformable_objects
            targets = [target[0] for target in targets]

            residuals_torch = []
            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_control_points(control_points)
            for i, target in enumerate(targets):
                self.exponential.set_initial_momenta(momenta[i])
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals_torch.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, target))

            residuals = np.zeros((self.number_of_objects,))
            for i in range(len(residuals_torch)):
                residuals += residuals_torch[i].detach().cpu().numpy()

            print("(initial ?) residuals", residuals)
            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k, obj in enumerate(self.objects_name):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(self.number_of_subjects)
                    self.objects_noise_variance[k] = nv
                    logger.info('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

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
        # self.number_of_control_points = len(cp)

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['momenta']

    def set_momenta(self, mom):
        self.fixed_effects['momenta'] = mom
    
    #ajout fg
    def get_coarse_momenta(self):
        return self.fixed_effects['coarse_momenta']
    
    def set_coarse_momenta(self, mom):
        self.fixed_effects['coarse_momenta'] = mom

    def get_coarse_points(self):
        return self.fixed_effects['coarse_points']
    
    def set_coarse_points(self, cp):
        self.fixed_effects['coarse_points'] = cp

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        if not self.freeze_momenta:
            out['momenta'] = self.fixed_effects['momenta']
        
        #ajout fg
        if self.optimize_nb_control_points:
            out['coarse_momenta'] = self.fixed_effects['coarse_momenta']
            if not self.freeze_control_points:
                out['coarse_points'] = self.fixed_effects['coarse_points']

        
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        if not self.freeze_momenta:
            self.set_momenta(fixed_effects['momenta'])
        #ajout fg
        if self.optimize_nb_control_points:
            self.set_coarse_momenta(fixed_effects['coarse_momenta'])
            if not self.freeze_control_points:
                self.set_coarse_points(fixed_effects['coarse_points'])


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def setup_multiprocess_pool(self, dataset):
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.deformable_objects],
                                                self.multi_object_attachment,
                                                self.objects_noise_variance,
                                                self.freeze_template, self.freeze_control_points, 
                                                self.freeze_momenta, 
                                                self.exponential, self.sobolev_kernel, self.use_sobolev_gradient,
                                                self.tensor_scalar_type, self.gpu_mode))

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        if self.number_of_processes > 1:
            targets = [target[0] for target in dataset.deformable_objects]
            args = [(i, self.template,
                     self.fixed_effects['template_data'],
                     self.fixed_effects['control_points'],
                     self.fixed_effects['momenta'][i],
                     with_grad) for i in range(len(targets))]

            start = time.perf_counter()
            results = self.pool.map(_subject_attachment_and_regularity, args, chunksize=1)  # TODO: optimized chunk size
            # results = self.pool.imap_unordered(_subject_attachment_and_regularity, args, chunksize=1)
            # results = self.pool.imap(_subject_attachment_and_regularity, args, chunksize=int(len(args)/self.number_of_processes))
            logger.debug('time taken for deformations : ' + str(time.perf_counter() - start))

            # Sum and return.
            if with_grad:
                attachment = 0.0
                regularity = 0.0
                #additione le gradient et l'attachement de chaque sujet
                gradient = {}
                if not self.freeze_template:
                    for key, value in self.fixed_effects['template_data'].items():
                        gradient[key] = np.zeros(value.shape)
                if not self.freeze_control_points:
                    gradient['control_points'] = np.zeros(self.fixed_effects['control_points'].shape)
                if not self.freeze_momenta:
                    gradient['momenta'] = np.zeros(self.fixed_effects['momenta'].shape)

                for result in results:
                    i, (attachment_i, regularity_i, gradient_i) = result
                    #ajout fg
                    i, (attachment_i, regularity_i, gradient_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                    #residus += residus_i #ajout fg
                    for key, value in gradient_i.items():
                        if key == 'momenta':
                            gradient[key][i] = value
                        else:
                            gradient[key] += value
                return attachment, regularity, gradient
            else:
                attachment = 0.0
                regularity = 0.0
                for result in results:
                    #i, (attachment_i, regularity_i) = result
                    #ajout fg
                    i, (attachment_i, regularity_i, gradient_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                return attachment, regularity

        else:
            device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, control_points, momenta, coarse_momenta = self._fixed_effects_to_torch_tensors(with_grad,
                                                                                                           device=device)
            return self._compute_attachment_and_regularity(dataset, template_data, template_points, control_points,
                                                           momenta, coarse_momenta, with_grad, device=device) 

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def _deform_and_compute_attachment_and_regularity(exponential, template_points, control_points, momenta,
                                                      template, template_data,
                                                      multi_object_attachment, deformable_objects,
                                                      objects_noise_variance,
                                                      device='cpu'):
        # Deform.
        exponential.set_initial_template_points(template_points)
        
        exponential.set_initial_control_points(control_points)
        exponential.set_initial_momenta(momenta)
        exponential.move_data_to_(device=device)
        exponential.update() #flow the template points according to the momenta using kernel.convolve

        # Compute attachment and regularity.
        #print("template_points", template_points['image_points'], template_points['image_points'].shape)
        deformed_points = exponential.get_template_points() #template points
        #print("deformed_points", deformed_points['image_points'], deformed_points['image_points'].shape)
        deformed_data = template.get_deformed_data(deformed_points, template_data) #template intensities after deformation
        #(observation) deformable multi object -> image -> torch.interpolate
        attachment = -multi_object_attachment.compute_weighted_distance(deformed_data, template, deformable_objects,
                                                                        objects_noise_variance)
        #print("attachment", attachment)
        regularity = -exponential.get_norm_squared()

        assert torch.device(
            device) == attachment.device == regularity.device, 'attachment and regularity tensors must be on the same device. ' \
                                                               'device=' + device + \
                                                               ', attachment.device=' + str(attachment.device) + \
                                                               ', regularity.device=' + str(regularity.device)
        
        return attachment, regularity
    
        

    @staticmethod
    def _compute_gradients(attachment, regularity, template_data,
                           freeze_template, template_points,
                           freeze_control_points, control_points,
                           freeze_momenta, momenta, 
                           optimize_nb_control_points, coarse_momenta, 
                           silent_coarse_momenta, zones,
                           dimension, points_per_axis, current_scale,
                           use_sobolev_gradient, sobolev_kernel,
                           with_grad=False):
        if with_grad:
            #coarse_momenta here = momenta, and they have the same gradients because of fixed effects to torch tensors
            total_for_subject = attachment + regularity #torch tensor
                        
            total_for_subject.backward() #compute gradient 
               
            gradient = {}
            if not freeze_template:
                if 'landmark_points' in template_data.keys():
                    assert template_points['landmark_points'].grad is not None, 'Gradients have not been computed'
                    if use_sobolev_gradient:
                        gradient['landmark_points'] = sobolev_kernel.convolve(
                            template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                            template_points['landmark_points'].grad.detach()).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    assert template_data['image_intensities'].grad is not None, 'Gradients have not been computed'
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
            if not freeze_control_points:
                assert control_points.grad is not None, 'Gradients have not been computed'
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if not freeze_momenta:
                assert momenta.grad is not None, 'Gradients have not been computed'
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()

            if optimize_nb_control_points:
                gradient = _compute_haar_transform_of_gradients(dimension, gradient, zones, current_scale, silent_coarse_momenta, points_per_axis, freeze_control_points)
                    
                print("\n compute_gradients")
                                                    
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient
        else:
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

        return res

    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, momenta, coarse_momenta,
                                            with_grad=False, device='cpu'):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output.
        Single-thread version.
        """

        # Initialize.
        targets = [target[0] for target in dataset.deformable_objects]
        attachment = 0.
        regularity = 0.
        #residus = torch.zeros((template_data['image_intensities'].shape), device=device, dtype=dtype) #ajout fg

        # loop for every deformable object
        # deform and update attachment and regularity
        for i, target in enumerate(targets):
            new_attachment, new_regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
                self.exponential, template_points, control_points, momenta[i],
                self.template, template_data, self.multi_object_attachment,
                target, self.objects_noise_variance,
                device=device)

            attachment += new_attachment
            regularity += new_regularity

        # Compute gradient.

        return self._compute_gradients(attachment, regularity, template_data,
                                       self.freeze_template, template_points,
                                       self.freeze_control_points, control_points,
                                       self.freeze_momenta, momenta, 
                                       self.optimize_nb_control_points, coarse_momenta,
                                       self.silent_coarse_momenta, self.zones, self.dimension, self.points_per_axis, self.current_scale,
                                       self.use_sobolev_gradient, self.sobolev_kernel,
                                       with_grad)

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                  requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}
        # template_data = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
        #                                requires_grad=(not self.freeze_template and with_grad))
        #                  for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                    requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}
        # template_points = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
        #                                  requires_grad=(not self.freeze_template and with_grad))
        #                    for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = utilities.move_data(control_points, device=device, dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_control_points and with_grad))
            # control_points = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type),
            #                           requires_grad=(not self.freeze_control_points and with_grad))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(not self.freeze_momenta and with_grad))
        # momenta = Variable(torch.from_numpy(momenta).type(self.tensor_scalar_type),
        #                    requires_grad=(not self.freeze_momenta and with_grad))

        #ajout fg
        coarse_momenta = self.fixed_effects['coarse_momenta']
        if self.optimize_nb_control_points:
            coarse_momenta = utilities.move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(self.optimize_nb_control_points and with_grad))


        return template_data, template_points, control_points, momenta, coarse_momenta #ajout fg

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, current_iteration, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals] #moyenne des résidus pour chaque sujet
            #ajout fg: moyenne des résidus
            mean_residuals = np.mean(np.asarray(residuals_list).flatten())
            mean_initial_residuals = np.sum(self.initial_residuals.flatten())
            #last_residuals_sum = np.sum(np.asarray(residuals_list).flatten()) #faux: trop élevé
            #residuals_ratio = 1 - last_residuals_sum/initial_residuals_sum
            residuals_ratio = 1 - mean_residuals/mean_initial_residuals


            residuals_list.append([0])
            residuals_list.append([mean_residuals])
            residuals_list.append([mean_initial_residuals]) #good
            #residuals_list.append([last_residuals_sum])
            residuals_list.append([residuals_ratio])

            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(output_dir, str(current_iteration))

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):
        device, _ = utilities.get_best_device(self.gpu_mode)

        # Initialize.
        template_data, template_points, control_points, momenta, _ = self._fixed_effects_to_torch_tensors(False, device=device)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            # Writing the whole flow. -> modif fg
            names = []
            for k, object_name in enumerate(self.objects_name):
                name = self.name + '__flow__' + object_name + '__subject_' + subject_id
                names.append(name)
            self.exponential.write_flow(names, self.objects_name_extension, self.template, template_data, output_dir) #silenced after
            
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
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, output_dir, current_iteration):

        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_"  + self.objects_name[i] + current_iteration + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_"  + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        if not self.freeze_control_points:
            write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints" + current_iteration + ".txt")
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        
        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta" + current_iteration + ".txt")
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")

        #ajout fg: write zones
        if self.optimize_nb_control_points:
            array = np.zeros((5000, 3))
            j = 0
            for scale in range(self.coarser_scale, max(self.current_scale -1, 0), -1):
                nombre_zones = len(self.zones[scale])
                nombre_zones_silencees = len(self.silent_coarse_momenta[scale])
                array[j] = scale, nombre_zones, nombre_zones_silencees
                for silent_zone in self.silent_coarse_momenta[scale]:
                    j += 1
                    array[j, 0] = silent_zone
                j += 2
            write_3D_array(array, output_dir, self.name + "_silenced_zones.txt")

        


    ####################################################################################################################
    ### Coarse to fine 
    ###ajouts fg
    ####################################################################################################################
    
    def compute_residuals(self, dataset, current_iteration, save_every_n_iters, output_dir):
        """
        Compute residuals at each pixel/voxel between objects and deformed template.
        Save a heatmap of the residuals
        """

        #print("template_data", template_data) #template_data['image_intensities]
        #print("template_points", template_points) #dico ['image_points]
        #print("self.fixed_effects.keys()", self.fixed_effects.keys())
        #print("new_parameters", new_parameters.keys())
        #print("gradient.keys()", gradient.keys())

        # Deform template
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, _ = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        
        #####compute residuals
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)
        residuals_by_point = torch.zeros((template_data['image_intensities'].shape), 
                                    device=next(iter(template_data.values())).device, 
                                    dtype=next(iter(template_data.values())).dtype)   #tensor not dict             

        for i, subject_id in enumerate(dataset.subject_ids):
            #deform template
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            deformed_points = self.exponential.get_template_points() #template points #tensor
            deformed_template = self.template.get_deformed_data(deformed_points, template_data) #dict containing tensor
            
            #get object intensities
            objet = dataset.deformable_objects[i][0]
            objet_intensities = objet.get_data()["image_intensities"]
            target_intensities = utilities.move_data(objet_intensities, device=next(iter(template_data.values())).device, 
                                    dtype = next(iter(template_data.values())).dtype) #tensor not dict 
            #compute residuals
            residuals_by_point += (1/dataset.number_of_subjects) * (target_intensities - deformed_template['image_intensities']) ** 2
        
        #residuals heat map
        if (not current_iteration % save_every_n_iters) or current_iteration in [0, 1]:
            names = "Heat_map_" + str(current_iteration) + self.objects_name_extension[0]
            deformed_template['image_intensities'] = residuals_by_point
            self.template.write(output_dir, [names], 
            {key: value.data.cpu().numpy() for key, value in deformed_template.items()})
        
        if current_iteration == 0:
            self.initial_residuals = residuals_by_point.cpu().numpy()
        
        return residuals_by_point.cpu().numpy()

    def save_residuals_by_zones(self, dataset, output_dir, current_iteration):
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, _ = \
            self._fixed_effects_to_torch_tensors(False, device = device)

        template_data['image_intensities'] = torch.zeros(template_data['image_intensities'].shape)

        if self.dimension == 3:
            
            #draw coarser zones that are silenced
            for scale in range(max([int(k) for k in self.silent_coarse_momenta.keys()]), self.current_scale, -1):
                print("scale", scale)
                print(self.current_scale)
                for zone in self.silent_coarse_momenta[scale]:
                    points_in_zone = self.zones[scale][zone]["points"]
                    for point_position in points_in_zone.tolist():
                        limits = [[max(0, math.floor(point_position[d] - self.deformation_kernel_width/2)), 
                                    min(template_data['image_intensities'].shape[d]-1, math.floor(point_position[d] + self.deformation_kernel_width/2))] \
                                        for d in range(self.dimension)]
                        template_data['image_intensities'][max(0, int(point_position[0]-1)):min(template_data['image_intensities'].shape[0], int(point_position[0]+1)),
                                                        max(0, int(point_position[1]-1)):min(template_data['image_intensities'].shape[1], int(point_position[1]+1)),
                                                        max(0, int(point_position[2]-1)):min(template_data['image_intensities'].shape[2], int(point_position[2]+1))] = 10000
                        
                        template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]] = (-1)*self.max_residus
                        
            
            #names = "Heat_map_scale_" + str(self.current_scale) + "_zones_" + str(0) + self.objects_name_extension[0]
            #self.template.write(output_dir, [names], {key: value.data.cpu().numpy() for key, value in template_data.items()})
            
            #draw finer zones
            for (zone, _) in self.zones[self.current_scale].items():
                residuals_in_zone = self.zones[self.current_scale][zone]["residuals"]
                points_in_zone = self.zones[self.current_scale][zone]["points"]
                if "coarser_zone" not in self.zones[self.current_scale][zone].keys():
                    if self.current_scale in self.silent_coarse_momenta.keys() and zone in self.silent_coarse_momenta[self.current_scale]:
                        for point_position in points_in_zone:
                            limits = [[max(0, int(point_position[d] - self.deformation_kernel_width/2)), 
                                        min(template_data['image_intensities'].shape[d]-1, int(point_position[d] + self.deformation_kernel_width/2))] \
                                            for d in range(self.dimension)]
                            #draw points
                            template_data['image_intensities'][max(0, int(point_position[0]-1)):min(template_data['image_intensities'].shape[0], int(point_position[0]+1)),
                                                            max(0, int(point_position[1]-1)):min(template_data['image_intensities'].shape[1], int(point_position[1]+1)),
                                                            max(0, int(point_position[2]-1)):min(template_data['image_intensities'].shape[2], int(point_position[2]+1))] = 10000
                            template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]] = (-0.5) * self.max_residus
                            
                    # else:
                    #     for point_position in points_in_zone.tolist():
                    #         limits = [[max(0, math.floor(point_position[d] - self.deformation_kernel_width/2)), 
                    #                     min(template_data['image_intensities'].shape[d], math.floor(point_position[d] + self.deformation_kernel_width/2))] \
                    #                         for d in range(self.dimension)]
                    #         template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]] = residuals_in_zone
                    #         template_data['image_intensities'][max(0, int(point_position[0]-1)):min(template_data['image_intensities'].shape[0], int(point_position[0]+1)),
                    #                                         max(0, int(point_position[1]-1)):min(template_data['image_intensities'].shape[1], int(point_position[1]+1)),
                    #                                         max(0, int(point_position[2]-1)):min(template_data['image_intensities'].shape[2], int(point_position[2]+1))] = 10000
            
            names = "Heat_map_scale_" + str(self.current_scale) + "_residuals" + "_zones_" + str(zone+1) + "iter_" + str(current_iteration) + self.objects_name_extension[0]
            self.template.write(output_dir, [names], {key: value.data.cpu().numpy() for key, value in template_data.items()})
        
        else:                        
            #draw finer zones
            for (zone, _) in self.zones[self.current_scale].items():
                residuals_in_zone = self.zones[self.current_scale][zone]["residuals"]
                points_in_zone = self.zones[self.current_scale][zone]["points"]
                if self.current_scale in self.silent_coarse_momenta.keys() and zone in self.silent_coarse_momenta[self.current_scale]:
                    for point_position in points_in_zone:
                        limits = [[max(0, int(point_position[d] - self.deformation_kernel_width/2)), 
                                    min(template_data['image_intensities'].shape[d]-1, int(point_position[d] + self.deformation_kernel_width/2))] \
                                        for d in range(self.dimension)]
                        #template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]] = self.max_residus*0.3

                        #delineate zone 
                        template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]] = self.max_residus
                        template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][1]] = self.max_residus
                        template_data['image_intensities'][limits[0][0], limits[1][0]:limits[1][1]] = self.max_residus
                        template_data['image_intensities'][limits[0][1], limits[1][0]:limits[1][1]] = self.max_residus
                    # else:
                    #     for point_position in points_in_zone.tolist():
                    #         limits = [[max(0, math.floor(point_position[d] - self.deformation_kernel_width/2)), 
                    #                     min(template_data['image_intensities'].shape[d], math.floor(point_position[d] + self.deformation_kernel_width/2))] \
                    #                         for d in range(self.dimension)]
                    #         template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]] = residuals_in_zone
            
            names = "Heat_map_scale_" + str(self.current_scale) + "_residuals" + "_zones_" + str(zone+1) + "iter_" + str(current_iteration) + self.objects_name_extension[0]
            self.template.write(output_dir, [names], {key: value.data.cpu().numpy() for key, value in template_data.items()})

        return

    def save_momenta_as_image(self, dataset, output_dir, momenta, control_points):
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, _ = \
            self._fixed_effects_to_torch_tensors(False, device = device)

        
        print("mom", momenta[0].shape, momenta.shape)
        for d in range(self.dimension):
            template_data['image_intensities'] = torch.zeros(template_data['image_intensities'].shape)
            for (p, point_position) in enumerate(control_points):
                limits = [[max(0, math.floor(point_position[d] - self.deformation_kernel_width/2)), 
                            min(template_data['image_intensities'].shape[d], math.floor(point_position[d] + self.deformation_kernel_width/2))] \
                                for d in range(self.dimension)]
                template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]] = momenta[0][p, d]
            
            names = "Heat_map_moments_" + str(self.current_scale) + "_sujet_" + str(0) + "_" + str(d) + self.objects_name_extension[0]
            self.template.write(output_dir, [names], {key: value.data.cpu().numpy() for key, value in template_data.items()})


    def save_deformation_field(self, dataset, output_dir):
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, _ = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        dimension = control_points.size(1)

        template_data['image_intensities'] = torch.zeros(template_data['image_intensities'].shape)

        control_points = torch.tensor(control_points, dtype = torch.float) #21 x 3
        momenta = torch.tensor(momenta, dtype = torch.float) #11 x 8 x 3

        #compute old vector field
        template_data['image_intensities'] = self.exponential.kernel.convolve(template_points, control_points, momenta[0])

        names = "Vector_field" + self.objects_name_extension[0]
        self.template.write(output_dir, [names], 
        {key: value.data.cpu().numpy() for key, value in template_data.items()})
                
        return

    
    def number_of_pixels(self, dataset):
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        number_of_pixels = 1
        for k in range(self.dimension):
            number_of_pixels = number_of_pixels * objet_intensities.shape[k]
        
        return number_of_pixels

    

    def save_points(self, current_iteration, control_points, dataset, output_dir):
        """
        Save control points on blank image
        """
        names = "Points_" + str(current_iteration) + "_" + str(len(control_points))+ self.objects_name_extension[0]
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        shape = tuple([objet_intensities.shape[k] for k in range(self.dimension)])
        new_objet = np.zeros(shape)
        for point in control_points:
            new_objet[tuple([int(t) for t in point])] = 1
        

        #add kernel width
        """
        if self.deformation_kernel_width.shape[0] == len(control_points):
            for (i,point) in enumerate(control_points):
                new_objet[tuple([int(t) for t in point])] = 1
                sigma = self.deformation_kernel_width[i]
        """

        self.template.write(output_dir, [names],  {"image_intensities":new_objet})

    def closest_neighbors(self, new_control_points, old_control_points):
        """
        Compute the 4 closest controls points for each point in new_control_points.
        """
        closest_points_list = []
        for point in new_control_points:
            distance_to_pts = [(np.sqrt(np.sum((point-old_control_points[k])**2, axis=0)), k) \
                for k in range(len(old_control_points))]                    
            #if the point existed before, we keep its momenta
            #else we average 4 closest neighbors
            closest_points_indices = [c[1] for c in sorted(distance_to_pts)[:3]]
            same_point = [d[1] for d in distance_to_pts if d[0] == 0]
            if len(same_point) > 0:
                closest_points_indices = same_point
            closest_points_list.append(closest_points_indices)
        
        return closest_points_list

    def set_new_momenta(self, new_control_points, old_momenta, closest_points_list):
        """
        Update the momenta for each new point and each subject
        The momenta of a new point =  average momenta of its 4 closest neighbors
        """
        new_moments = np.zeros((old_momenta.shape[0], new_control_points.shape[0],
                                new_control_points.shape[1])) #n sujets x n points x dimension
        #for each subject -> for each new point -> add a new momenta
        for ind, old_momenta_subject in enumerate(old_momenta):
            new_momenta_subject = np.zeros((new_control_points.shape[0],new_control_points.shape[1]))

            for (i,new_point) in enumerate(new_control_points):            
                #average of closest neighbours momenta
                new_coordinates = []
                for k in range(self.dimension):
                    new_coordinates.append(np.mean([old_momenta_subject[c][k] for c in closest_points_list[i]]))
                new_momenta_subject[i] = np.asarray(new_coordinates)

            new_moments[ind] = new_momenta_subject
        
        return new_moments
    
    def compute_new_vector_field(self, old_control_points, new_control_points, old_momenta, new_kernel_width):
        """
        Update momenta values by resolving system of equations to preserve vector field
        Old moments of each new ctrl points = coefficients (convolution between new cp) x new moments
        """
        #print("old kernel", self.deformation_kernel_width)
        new_control_points = torch.tensor(new_control_points, dtype = torch.float) #21 x 3
        old_control_points = torch.tensor(old_control_points, dtype = torch.float) #8x3
        old_momenta = torch.tensor(old_momenta, dtype = torch.float) #11 x 8 x 3

        #compute old vector field
        old_vect_field = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2]))
        for ind in range(old_momenta.shape[0]):
            vect_field = self.exponential.kernel.convolve(new_control_points, old_control_points, old_momenta[ind])
            old_vect_field[ind] = vect_field.cpu().numpy()

        #print("old_vect_field", old_vect_field.shape, old_vect_field[0])
        
        #compute coefficient (kernel convolution between new control points) (same for everyone)
        coef_new_vect_field = np.zeros((len(new_control_points), len(new_control_points)))
        for (i, point_i) in enumerate(new_control_points):
            for (j, point_j) in enumerate(new_control_points):
                square_distance = np.linalg.norm(point_i-point_j)**2
                coefficient = np.exp((-1/new_kernel_width[j]** 2) * square_distance) 
                coef_new_vect_field[i, j] = coefficient
        #print("coef_new_vect_field", coef_new_vect_field)

        #solve equations
        new_momenta = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2])) 
        for ind in range(old_momenta.shape[0]):
            x = np.linalg.solve(coef_new_vect_field, old_vect_field[ind]) #AX = B solve(A, B) -> B old moments, A new moments B (len(new ctl points)) A 
            new_momenta[ind] = x
            #print("old_momenta", old_momenta[ind])
            #print("new_momenta", x)

        #print("new_momenta", new_momenta)
        
        return new_momenta


    def save_new_parameters(self, new_parameters, new_moments, new_coarse_momenta, new_control_points):
        """
            Save new control points and momenta in the model.
        """
        self.set_control_points(new_control_points)
        new_parameters['momenta'] = new_moments
        new_parameters['coarse_momenta'] = new_coarse_momenta
        fixed_effects = {key: new_parameters[key] for key in self.get_fixed_effects().keys()}
        self.set_fixed_effects(fixed_effects)

        return new_parameters

    def compute_new_kernel_width(self, nb_points_origin = None):
        """
            Compute new kernels for each point. Width = distance to closest neighbor.
        """
        control_points = self.fixed_effects['control_points']
        new_kernel_width = np.full((control_points.shape[0], 1), 5)
        for (i, point) in enumerate(control_points):
            if nb_points_origin is not None and i < nb_points_origin:
                if isinstance(self.deformation_kernel_width, int):
                    new_kernel_width[i] = int(self.deformation_kernel_width)
                else:
                    new_kernel_width[i] = int(self.deformation_kernel_width[i])
            else:
                distance_to_pts = [(np.sqrt(np.sum((point-control_points[k])**2, axis=0))) \
                    for k in range(len(control_points)) if (k != i)]
                min_dist_to_another_point = [c for c in sorted(distance_to_pts) if c != 0][0]
                new_kernel_width[i] = max(int(min_dist_to_another_point), 1)
                
        
        print("new_kernel_width", new_kernel_width.shape)
        #print("new_kernel_width", new_kernel_width)
        return new_kernel_width

    def adapt_kernel(self, new_kernel_width):
        """
            Save new kernel width
        """
        self.deformation_kernel_width = new_kernel_width
        
        self.exponential = Exponential(
            dense_mode=self.dense_mode,
            kernel=kernel_factory.factory(self.deformation_kernel_type,
                                        gpu_mode=self.gpu_mode,
                                        kernel_width=new_kernel_width),
            shoot_kernel_type=self.exponential.shoot_kernel_type,
            number_of_time_points=self.exponential.number_of_time_points,
            use_rk2_for_shoot=self.exponential.use_rk2_for_shoot, 
            use_rk2_for_flow=self.exponential.use_rk2_for_flow)
    
    def residus_moyens_voisins(self, coord, new_spacing, residuals_by_point, objet_intensities):
        if self.dimension == 2:
            limite_inf1 = max(coord[0]-int(new_spacing/2), 0)
            limite_sup1 = min(coord[0]+ int(new_spacing/2) + 1, objet_intensities.shape[0])
            limite_inf2 = max(coord[1]-int(new_spacing/2), 0)
            limite_sup2 = min(coord[1]+ int(new_spacing/2) + 1, objet_intensities.shape[1])
            zone = residuals_by_point[limite_inf1:limite_sup1, limite_inf2:limite_sup2]
        elif self.dimension == 3:
            limite_inf1 = max(coord[0]-int(new_spacing/2), 0)
            limite_sup1 = min(coord[0]+ int(new_spacing/2) + 1, objet_intensities.shape[0])
            limite_inf2 = max(coord[1]-int(new_spacing/2), 0)
            limite_sup2 = min(coord[1]+ int(new_spacing/2) + 1, objet_intensities.shape[1])
            limite_inf3 = max(coord[2]-int(new_spacing/2), 0)
            limite_sup3 = min(coord[2]+ int(new_spacing/2) + 1, objet_intensities.shape[2])
            zone = residuals_by_point[limite_inf1:limite_sup1, limite_inf2:limite_sup2, limite_inf3:limite_sup3]
        
        return np.mean(zone)

    def residus_moyens_voisins2(self, residuals_by_point, output_dir):
        """
        Gaussian filter of the residuals to account for neighborhood
        """
        dimensions = (residuals_by_point.shape)
        voxels = np.zeros((int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), 3))
        residuals_by_point2 = np.zeros((int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), 3))
        for i in range(int(dimensions[0])):
            for j in range(int(dimensions[1])):
                for k in range(int(dimensions[2])):
                    voxels[i, j, k] = [i, j, k]
                    residuals_by_point2[i, j, k, :] = residuals_by_point[i, j, k]
        voxels = torch.tensor(voxels, dtype = torch.float)
        residuals_by_point2 = torch.tensor(residuals_by_point2, dtype = torch.float)

        old_kernel_width = self.deformation_kernel_width
        self.adapt_kernel(new_kernel_width = 1.5)
        #/!\ convolution prévue pour des vecteurs de dimension 3 et non pas 1 -> redimensionner
        convole_res = self.exponential.kernel.convolve(voxels.contiguous().view(-1, 3), voxels.contiguous().view(-1, 3), #voxels x 3
                                                    residuals_by_point2.contiguous().view(-1, 3)) #voxels x 3
        convole_res = convole_res.contiguous().view(residuals_by_point.shape[0], residuals_by_point.shape[1], residuals_by_point.shape[2], 3).cpu().numpy()
        convole_res = convole_res[:, :, :, 0]
        #print("convole_res", convole_res.shape, convole_res[50, 30])
        #print("residuals_by_point", residuals_by_point[50, 30])
        self.adapt_kernel(new_kernel_width = old_kernel_width)

        #save residus
        dico = {}
        dico['image_intensities'] = torch.tensor(convole_res, dtype = torch.float)
        self.template.write(output_dir, ["Heat_map_test.nii"], {key: value.data.cpu().numpy() for key, value in dico.items()})

        return convole_res
    
    def update_vector_field(self, new_spacing, old_control_points, old_momenta, new_control_points, silent_coarse_momenta = []):
        """
            Given old_control_points and old_momenta and new_control_points,
            Computes the vector field values at the new_control_points
            Then calculates the new momenta values at the new control points that preserve the vector field
            v_old = K_old alpha = K M B

            silent_coarse_momenta: indices of moment that must be set to 0            
        """
        new_control_points = torch.tensor(new_control_points, dtype = torch.float) #21 x 3
        old_control_points = torch.tensor(old_control_points, dtype = torch.float) #8x3
        old_momenta = torch.tensor(old_momenta, dtype = torch.float) #11 x 8 x 3

        #compute old vector field values at the new control points (using old kernel)
        old_vect_field = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2]))
        for sujet in range(old_momenta.shape[0]):
            vect_field = self.exponential.kernel.convolve(new_control_points, old_control_points, old_momenta[sujet])
            old_vect_field[sujet] = vect_field.cpu().numpy()

        print("old_vect_field", old_vect_field.shape, old_vect_field[0])
        
        #compute coefficient (kernel convolution between new control points using new kernel)
        new_kernel_width = new_spacing
        coef_new_vect_field = np.zeros((len(new_control_points), len(new_control_points)))
        for (i, point_i) in enumerate(new_control_points):
            for (j, point_j) in enumerate(new_control_points):
                if i not in silent_coarse_momenta and j not in silent_coarse_momenta:
                    if coef_new_vect_field[i, j] == 0:
                        square_distance = np.linalg.norm(point_i-point_j)**2
                        coefficient = np.exp((-1/new_kernel_width** 2) * square_distance) 
                        coef_new_vect_field[i, j], coef_new_vect_field[j, i] = coefficient, coefficient
        print("coef_new_vect_field", coef_new_vect_field.shape, coef_new_vect_field)

        #multiply these coefficients by the new haar matrix
        coef_new_vect_field = np.matmul(coef_new_vect_field, self.haar_matrix)

        #solve equations
        new_coarse_momenta = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2])) 
        for sujet in range(old_momenta.shape[0]):
            x = np.linalg.solve(coef_new_vect_field, old_vect_field[sujet]) #AX = B solve(A, B) -> B old moments, A new moments B (len(new ctl points)) A 
            new_coarse_momenta[sujet] = x

        return new_coarse_momenta

    def naive_coarse_to_fine(self, new_parameters, current_iteration, output_dir, dataset):
            
            print("\n Naive coarse to fine")

            #get control points
            device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, old_control_points, old_momenta, old_coarse_momenta = \
                self._fixed_effects_to_torch_tensors(False, device = device)
            
            old_control_points = old_control_points.cpu().numpy()
            old_momenta = old_momenta.cpu().numpy()
            old_coarse_momenta = old_coarse_momenta.cpu().numpy()

            #add new control points
            #total number of points must be multiple of power 2!
            
            #new_spacing, new_control_points = self.add_points_ctf(self.deformation_kernel_width, current_iteration, len(old_control_points))
            new_spacing, new_control_points = self.add_points_regularly(current_iteration)

            #keep old points
            print("len(new_control_points)", len(new_control_points))
            #new_control_points = np.concatenate((old_control_points, new_control_points)) 

            if (self.dimension == 2 and new_spacing > 1) or (self.dimension == 3 and new_spacing > self.max_spacing):
                #save new points
                self.save_points(current_iteration, new_control_points, dataset, output_dir)

                #self.haar_matrix = haarMatrix2(len(new_control_points))
                print("new_haar_matrix 2",  np.shape(self.haar_matrix), self.haar_matrix)
                
                #test with identity
                self.haar_matrix = np.eye(len(new_control_points))

                #update COARSE momenta by preserving vector field
                new_coarse_momenta = self.update_vector_field(new_spacing, old_control_points, old_momenta, new_control_points)
                new_momenta = np.matmul(self.haar_matrix, new_coarse_momenta)

                #save new parameters            
                new_parameters = self.save_new_parameters(new_parameters, new_momenta, new_coarse_momenta, new_control_points)
                print("after adding points", np.shape(self.fixed_effects['control_points']),
                np.shape(self.fixed_effects['momenta']), np.shape(self.fixed_effects['coarse_momenta']))

                print("old_momenta[0]", old_momenta[0])
                print("self.fixed_effects['momenta'][0]", self.fixed_effects['momenta'][0])
                print("self.fixed_effects['coarse_momenta'][0]", self.fixed_effects['coarse_momenta'][0])

                #save new kernel
                self.adapt_kernel(new_kernel_width = new_spacing)  

            return new_parameters


    def naive_coarse_to_fine2(self, new_parameters, current_iteration, output_dir, dataset):
            
        print("\n Naive coarse to fine 2")

        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, old_control_points, old_momenta, old_coarse_momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        
        old_control_points = old_control_points.cpu().numpy()
        old_momenta = old_momenta.cpu().numpy()
        old_coarse_momenta = old_coarse_momenta.cpu().numpy()            
        
        #we do not add points
        #nor update vector field 
        #nor adapt kernel
        self.current_scale += 1 
        print("self.current_scale", self.current_scale)

        #silence coefficients 


        return new_parameters

    def compute_wavelets_position_at_current_scale(self, coarse_momenta):
        """
        Compute a list of all wavelets positions at the current scale
        """
        list_wavelet_positions = []
        if self.dimension == 3:
            for x in range(coarse_momenta[0][0].wc.shape[0]): #browse shape of cp
                for y in range(coarse_momenta[0][0].wc.shape[1]):
                    for z in range(coarse_momenta[0][0].wc.shape[2]):
                        position, _, scale = coarse_momenta[0][0].haar_coeff_pos_type((x, y, z))
                        if scale == self.current_scale and position not in list_wavelet_positions:
                            list_wavelet_positions.append(position)
        else:
            for x in range(coarse_momenta[0][0].wc.shape[0]): #browse shape of cp
                for y in range(coarse_momenta[0][0].wc.shape[1]):
                    position, _, scale = coarse_momenta[0][0].haar_coeff_pos_type((x, y))
                    if scale == self.current_scale and position not in list_wavelet_positions:
                        list_wavelet_positions.append(position)

        #in coarse scale, we can miss the small zone at the corner, which has no dd, ad, or da (and of course no aa)
        #no importance in the variability (it will never be divided)
        
        print("list_wavelet_positions", list_wavelet_positions)

        return list_wavelet_positions
    
    def compute_wavelet_size(self):
        wavelets_size = [2** self.current_scale for d in range(self.dimension)]

        check = [w <= x for (w, x) in zip(wavelets_size, self.points_per_axis)]
        
        if False in check:
            wavelets_size = self.points_per_axis
        print("wavelets_size", wavelets_size)

        return wavelets_size
    

    def compute_voxels_position_on_image(self, residuals):
        #array of np.indices = self.dim x shape_1, x shape 2 (x shape 3) -> array[:, x, y, z] = [x, y, z]
        if self.dimension == 3:
            voxels_pos_image = np.indices((residuals.shape[0], residuals.shape[1], residuals.shape[2])).transpose((1,2,3,0))
        else:
            voxels_pos_image = np.indices((residuals.shape[0], residuals.shape[1])).transpose((1,2,0))
        
        #nb_of_voxels = np.product([residuals.shape[k] for k in range(self.dimension)])

        return voxels_pos_image

    def fetch_control_points_in_zone(self, position, limites, control_points):
        """
            Fetch coordinates of the controls points in a specific zone
        """ 

        if self.dimension == 3:
            points_in_zone = control_points[position[0]:limites[0], position[1]:limites[1], position[2]:limites[2], :]
        else:
            points_in_zone = control_points[position[0]:limites[0], position[1]:limites[1], :]
        
        #reshape points in zone for easy browsing : n_cp x dim
        points_in_zone = np.reshape(points_in_zone, (np.product(points_in_zone.shape[:-1]), points_in_zone.shape[-1]))

        return points_in_zone
    
    def residuals_around_point(self, point_position, residuals, voxels_pos_image):
        """
            Given a point position,  
        """
        #limits of the zone around point (in voxels coordinates)
        limits = [[max(0, int(point_position[d] - self.deformation_kernel_width)), 
                    min(residuals.shape[d], int(point_position[d] + self.deformation_kernel_width))] for d in range(self.dimension)]
        
        #voxels positions around point
        if self.dimension == 3:
            voxels_pos = voxels_pos_image[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
        else:
            voxels_pos = voxels_pos_image[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]]

        voxels_pos = np.reshape(voxels_pos, (np.product(voxels_pos.shape[:-1]), self.dimension)) #n voxels x 3
        voxels_pos = torch.tensor(voxels_pos, dtype = torch.float)

        #residuals around point
        if self.dimension == 3:
            voxels_res = residuals[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
        else:
            voxels_res = residuals[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]]
        
        voxels_res = np.reshape(voxels_res.flatten(), (len(voxels_res.flatten()), 1))
        voxels_res = np.concatenate(tuple([voxels_res for k in range(self.dimension)]), axis = 1) #n voxels x 3
        voxels_res = torch.tensor(voxels_res, dtype = torch.float)

        point_position = torch.tensor(np.reshape(point_position, (1, self.dimension)), dtype = torch.float) 
        
        #convolve residuals around point
        residuals_around_point = self.exponential.kernel.convolve(point_position, voxels_pos, voxels_res).cpu().numpy()[0][0]
        
        return residuals_around_point

    def store_zone_information(self, zone, position, wavelets_size, control_points, voxels_pos_image, residuals):
        """
            Store information about a specific zone in a dict
        """           
        #fetch control points in zone - position of cp in voxels coordinates
        limites = [min(position[d] + wavelets_size[d], control_points.shape[d]) for d in range(self.dimension)]
        points_in_zone = self.fetch_control_points_in_zone(position, limites, control_points) 

        #compute residuals in zone
        residuals_in_zone = [self.residuals_around_point(point_position, residuals, voxels_pos_image) for point_position in points_in_zone]
        
        #compute nb of voxels
        number_voxels_in_zone = np.product([limites[d]-position[d] for d in range(self.dimension)])

        self.zones[self.current_scale][zone] = dict()
        self.zones[self.current_scale][zone]["position"] = position
        self.zones[self.current_scale][zone]["points"] = points_in_zone
        self.zones[self.current_scale][zone]["size"] = wavelets_size
        self.zones[self.current_scale][zone]["residuals"] = sum(residuals_in_zone) 
        self.zones[self.current_scale][zone]["residuals_ratio"] = sum(residuals_in_zone)/number_voxels_in_zone

    def compute_maximum_residuals(self, list_wavelet_positions):
        self.max_residus = max([self.zones[self.current_scale][z]["residuals"] for z, _ in enumerate(list_wavelet_positions)])
        print("self.max_residus", self.max_residus)

    def silence_zones_in_silent_coarser_zone(self, zone):
        """
            Silence zones belonging to a coarser zone that was already silenced in previous coarse to fine iteration
        """
        points_in_zone = self.zones[self.current_scale][zone]["points"]
                        
        if self.current_scale+1 < self.coarser_scale:
            for coarse_zone in self.silent_coarse_momenta[self.current_scale+1]:
                points_in_coarse_zone = self.zones[self.current_scale+1][coarse_zone]["points"]
                pos = self.zones[self.current_scale+1][coarse_zone]["position"]
                common_points = [point for point in points_in_zone.tolist() if point in points_in_coarse_zone.tolist()]
                
                if common_points != []:
                    self.silent_coarse_momenta[self.current_scale].append(zone)
                    self.zones[self.current_scale][zone]["coarser_zone"] = coarse_zone
                    self.sum_already_silent += 1

    def silence_smooth_zones(self, zone):
        """
            Silence zones with low residuals
        """
        if self.zones[self.current_scale][zone]["residuals"] < 0.01*self.max_residus \
        and zone not in self.silent_coarse_momenta[self.current_scale]:
            self.silent_coarse_momenta[self.current_scale].append(zone)
    
    def coarse_to_fine_condition(self, current_iteration, avg_residuals):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        residuals_gain = (avg_residuals[-2] - avg_residuals[-1])/avg_residuals[-2]

        return ((self.iterations_coarse_to_fine == [] and current_iteration >= 3) or \
                (self.iterations_coarse_to_fine != [] and current_iteration - self.iterations_coarse_to_fine[-1] > 2))\
                and (residuals_gain < 0.01)

    def coarse_to_fine_images_condition(self, current_iteration, avg_residuals):
        print("coarse_to_fine_images_condition")
        if current_iteration == 0:
            return True

        print(current_iteration)
        if current_iteration > 1:
            residuals_gain = (avg_residuals[-2] - avg_residuals[-1])/avg_residuals[-2]
            print(residuals_gain)
            print(self.current_image_scale)
            print(self.iterations_coarse_to_fine)
            return ((residuals_gain < 0.02 and self.current_image_scale > 0) \
                and (current_iteration-self.iterations_coarse_to_fine[-1] > 5))
        
        return False
            
            #current_iteration-self.iterations_coarse_to_fine[-1] > 5)) #desynchronisation des deux CTF
            #or (self.iterations_coarse_to_fine != [] and self.iterations_coarse_to_fine[-1] == current_iteration)\
            #or (self.current_scale == 0 and current_iteration-self.iterations_coarse_to_fine[-1] > 2))

    def coarse_to_fine(self, new_parameters, current_iteration, output_dir, dataset, residuals, avg_residuals, naive = False):
        
        if self.coarse_to_fine_condition(current_iteration, avg_residuals):

            print("\n Coarse to fine")
            print("current_iteration", current_iteration)
            #go to smaller scale
            #beginning : self.current_scale = J
            self.current_scale = max(0, self.current_scale-1)
            print("self.current_scale", self.current_scale)
            #we only go to scale 0 so that the local adaptation is not called again ... 

            if self.current_scale in range(1, self.coarser_scale):
                self.iterations_coarse_to_fine.append(current_iteration)
                #get control points
                device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
                template_data, template_points, control_points, momenta, coarse_momenta = \
                    self._fixed_effects_to_torch_tensors(False, device = device)

                control_points = np.reshape(control_points.cpu().numpy(), tuple(self.points_per_axis + [self.dimension]))
                coarse_momenta = new_parameters["coarse_momenta"]
                
                #define smooth zones
                print("search smooth zones ...")
                print("self.current_scale", self.current_scale, "max (coarser) scale", self.coarser_scale)
                print("coarse_momenta.ws", coarse_momenta[0][0].ws)

                #compute size : number of points in biggest wavelet
                wavelets_size = self.compute_wavelet_size()

                #compute list of unique wavelet positions
                list_wavelet_positions = self.compute_wavelets_position_at_current_scale(coarse_momenta)
            
                #compute voxels positions on image
                voxels_pos_image = self.compute_voxels_position_on_image(residuals)
                
                self.zones[self.current_scale] = dict()
                for (zone, position) in enumerate(list_wavelet_positions):
                    self.store_zone_information(zone, position, wavelets_size, control_points, voxels_pos_image, residuals)               

                #silence smooth zones  
                self.compute_maximum_residuals(list_wavelet_positions)

                self.silent_coarse_momenta[self.current_scale] = []
                self.sum_already_silent = 0

                if not naive:
                    for (zone, position) in enumerate(list_wavelet_positions):
                        self.silence_zones_in_silent_coarser_zone(zone)
                        self.silence_smooth_zones(zone)
                    
                self.save_residuals_by_zones(dataset, output_dir, current_iteration)
                print("self.silent_coarse_momenta", self.silent_coarse_momenta)                      
                print("Out of", len(self.zones[self.current_scale].keys()), "zones", "silence", len(self.silent_coarse_momenta[self.current_scale]))
                print(self.sum_already_silent, "were already silenced by coarser zone")
            
    
    def compute_filter_width(self):
        wavelet_size = self.compute_wavelet_size() #nb of points in zone along each dimension
        proportion_of_zone = [w/p for (w, p) in zip(wavelet_size, self.points_per_axis)]

        sigma = [r * self.shape[i] for (i,r) in enumerate(proportion_of_zone)]
        sigma = int(np.mean(sigma)/20)

        print("sigma", sigma)
        sigma = 2
        return sigma 

    def save_image(self, intensities, output_dir, names = None):
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        image_data, template_points, control_points, momenta, _ = \
            self._fixed_effects_to_torch_tensors(False, device = device)

        image_data['image_intensities'] = intensities

        if not names:
            names = "Subject_" + str(self.current_image_scale) + self.objects_name_extension[0]
        else:
            names = names + str(self.current_image_scale) + self.objects_name_extension[0]

        self.template.write(output_dir, [names], {key: value for key, value in image_data.items()})

    

    def coarse_to_fine_on_images(self, original_dataset, current_dataset, current_iteration, output_dir, avg_residuals):
        
        #check that CTF happened before            
        if  self.coarse_to_fine_images_condition(current_iteration, avg_residuals):
            
            print("\n Coarse to fine on images")
            new_dataset = copy.deepcopy(original_dataset)

            if current_iteration != 0:
                self.current_image_scale = max(0, self.current_image_scale-1)
            print("sigma", self.current_image_scale)

            if current_iteration not in self.iterations_coarse_to_fine:
                self.iterations_coarse_to_fine.append(current_iteration)

            #filter images
            for i, subject_id in enumerate(new_dataset.subject_ids):
                for e in new_dataset.deformable_objects[i][0].object_list:
                    intensities = gaussian_filter(e.get_intensities(), sigma = self.current_image_scale)
                    e.set_intensities(intensities)
            
            #save the last images
            self.save_image(intensities, output_dir, names = "last_subject_" + str(current_iteration) + "_")

            return new_dataset
        
        return current_dataset
    
    def difference_scales_points_voxels(self):
        """Difference between maximum scale on images and maximum scale on momenta
        """
        intensities = self.template.get_data()["image_intensities"]
        maximum_scale = haar_forward(intensities).J
        difference_scales_points_voxels = maximum_scale - self.coarser_scale
        print("difference_scales_points_voxels", difference_scales_points_voxels)
        print("maximum_scale for images", maximum_scale)
        print("maximum_scale for points", self.coarser_scale)

        return difference_scales_points_voxels

    def haar_transform_and_filter_intensities(self, intensities):
        """ Haar transform a subject intensities according to the current image scale
        """
        intensities_haar = haar_forward(intensities, J = self.current_image_scale)
        #aa at current scale + ad, dd, da
        indices_to_browse_along_dim = [list(range(e)) for e in list(intensities_haar.wc.shape)]
        for indices in itertools.product(*indices_to_browse_along_dim):
            position, type, scale = intensities_haar.haar_coeff_pos_type([i for i in indices])
            #silence finer zones that we haven't reached yet
            if scale < self.current_image_scale and type != ['L', 'L']:
                intensities_haar.wc[indices] = 0

        intensities = intensities_haar.haar_inverse()

        return intensities

    def coarse_to_fine_on_images_haar(self, original_dataset, current_dataset, current_iteration, output_dir, avg_residuals):
        """
        Apply Haar filter on each subject image according to the new scale
        """

        #check that momenta CTF happened before
        if self.coarse_to_fine_images_condition(current_iteration):
            
            print("\n Coarse to fine on images")
            new_dataset = copy.deepcopy(original_dataset)

            if current_iteration != 0:
                self.current_image_scale = max(0, self.current_image_scale - 1)
                if self.current_image_scale == 1:
                    self.current_image_scale = 0

            if current_iteration not in self.iterations_coarse_to_fine:
                self.iterations_coarse_to_fine.append(current_iteration)            
            
            print("self.current_image_scale", self.current_image_scale)
            for i, subject_id in enumerate(new_dataset.subject_ids):
                for e in new_dataset.deformable_objects[i][0].object_list:
                    intensities = self.haar_transform_and_filter_intensities(e.get_intensities())
                    e.set_intensities(intensities)
                print("done with subject", i)

            #save the last images
            self.save_image(intensities, output_dir, names = "last_subject_" + str(current_iteration) + "_")
        
            return new_dataset

        return current_dataset


        """
        count = 0
        wavelet_size_voxels = [w*residuals.shape[d]/self.points_per_axis[d] for (d,w) in enumerate(wavelet_current_size)]
        print("wavelet_size_voxels", wavelet_size_voxels)
        
        zones = []
        for i in range(0, residuals.shape[0], int(math.ceil(wavelet_size_voxels[0]))):
            for j in range(0, residuals.shape[1], int(math.ceil(wavelet_size_voxels[1]))):
                for k in range(0, residuals.shape[2], int(math.ceil(wavelet_size_voxels[2]))):
                    print("zone", i, j, k)
                    limit_sup_i = min(i + int(wavelet_size_voxels[0]), residuals.shape[0]) 
                    limit_sup_j = min(j + int(wavelet_size_voxels[1]), residuals.shape[1]) 
                    limit_sup_k = min(k + int(wavelet_size_voxels[2]), residuals.shape[2]) 
                    print(limit_sup_i, limit_sup_j, limit_sup_k)
                    zones.append(([i, j, k], np.sum(residuals[i:limit_sup_i, j:limit_sup_j, k:limit_sup_k])))
        print("zones", zones)"""
        """
        zones = []
        for i in range(0, self.points_per_axis[0], wavelet_current_size[0]):
            for j in range(0, self.points_per_axis[1], wavelet_current_size[1]):
                for k in range(0, self.points_per_axis[2], wavelet_current_size[2]):
                    limit_inf_i = int(math.ceil(i*residuals.shape[0]/self.points_per_axis[0]))
                    limit_inf_j = int(math.ceil(j*residuals.shape[1]/self.points_per_axis[1]))
                    limit_inf_k = int(math.ceil(k*residuals.shape[2]/self.points_per_axis[2]))
                    limit_sup_i = min(limit_inf_i + int(wavelet_size_voxels[0]), residuals.shape[0]) 
                    limit_sup_j = min(limit_inf_j + int(wavelet_size_voxels[1]), residuals.shape[1]) 
                    limit_sup_k = min(limit_inf_k + int(wavelet_size_voxels[2]), residuals.shape[2]) 
                    print("zone", limit_inf_i, limit_inf_j, limit_inf_k)
                    print(limit_sup_i, limit_sup_j, limit_sup_k)
                    zones.append(([i, j, k], np.sum(residuals[limit_inf_i:limit_sup_i, limit_inf_j:limit_sup_j, limit_inf_k:limit_sup_k])))
        
        print("zones", zones)
        print("nb zones", len(zones))
        print(self.points_per_axis)
        print("residuals.shape", residuals.shape)
        print("wavelet_current_size", wavelet_current_size)
        print("new_parameters['coarse_momenta'][0][0].ws", new_parameters['coarse_momenta'][0][0].ws)
        zones_to_silence = [z[0] for z in zones if z[1] < 10]
        self.silent_coarse_momenta[self.current_scale] = zones_to_silence
        print("self.silent_coarse_momenta", self.silent_coarse_momenta)

        #just for check
        for i in range(new_parameters['coarse_momenta'][0][0].wc.shape[0]):
            for j in range(new_parameters['coarse_momenta'][0][0].wc.shape[1]):
                for k in range(new_parameters['coarse_momenta'][0][0].wc.shape[2]):
                    print(new_parameters['coarse_momenta'][0][0].haar_coeff_pos_type((i,j,k)))"""


        #save new points
        #self.save_points(current_iteration, new_control_points, dataset, output_dir)
        
        #define smooth zones
        #browse coefficient in the new scale -> how many points = 1 coef in the new scale ?
        """
        coarse_momenta_sub0_axis0 = coarse_momenta[0][0]
        print("coarse_momenta", coarse_momenta)
        print("coarse_momenta_sub0_axis0", coarse_momenta_sub0_axis0)
        coefficients_in_new_scale = coarse_momenta_sub0_axis0[self.current_scale]
        print("coefficients_in_new_scale", coefficients_in_new_scale)
        nb_coef = len(coefficients_in_new_scale["ddd"].flatten())
        print("nb_coef", nb_coef)
        nb_points_per_coef = len(control_points)/nb_coef
        print("nb_control_points_per_coef", nb_points_per_coef)
        print("a coef covers", nb_points_per_coef/len(control_points), "of the image")

        self.silent_coarse_momenta += coefficients_in_new_scale
        #for type in coefficients_in_new_scale.keys():
        type = "ddd"
        coef_to_points = dict()
        for x in range(coefficients_in_new_scale[type].shape[0]):
            for y in range(coefficients_in_new_scale[type].shape[1]):
                for z in range(coefficients_in_new_scale[type].shape[2]):
                    all_corresponding_points = []
                    all_corresponding_points2 = []
                    #not the point coordinate - point position in 3D matrix
                    coord_x = x*self.points_per_axis[0] * 2**((-1)*self.current_scale)
                    coord_y = y*self.points_per_axis[1] * 2**((-1)*self.current_scale)
                    coord_z = z*self.points_per_axis[2] * 2**((-1)*self.current_scale)

                    corresponding_point = [x*self.points_per_axis[0] * 2**((-1)*self.current_scale),  #x nx * 2^-j
                                            y*self.points_per_axis[1] * 2**((-1)*self.current_scale),
                                            z*self.points_per_axis[2] * 2**((-1)*self.current_scale)]
                    
                    print("corresponding_point", x, y, z, corresponding_point)
                    #x, y, z <=> z + n1 * x + n1 * n2 * y 
                    position_flatten_array = corresponding_point[0] + corresponding_point[1] * self.points_per_axis[0] \
                                            + corresponding_point[2] * self.points_per_axis[0] * self.points_per_axis[1]
                    print("position in flatten_array", position_flatten_array)

                    all_corresponding_points.append(corresponding_point)
                    all_corresponding_points2.append(position_flatten_array)

                    #compute residuals in the zone
                    position_in_image = control_points[position_flatten_array]
                    print("position_in_image", position_in_image)
                    len(control_points)/nb_points_per_coef
                    intensities = self.template.get_data()["image_intensities"]
                    width = min([intensities.shape[k] for k in range(self.dimension)])
                    zone = largeur/nb_coef ** (1/3)
                    limits = [(corresponding_point[d], corresponding_point[d] + zone[d])  for d in range(self.dimension)]
                    residuals_around_point = residuals[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
                    residuals_around_points.append(np.sum(residuals_around_point))


                    # k = 1
                    # while len(all_corresponding_points) < nb_points_per_coef:
                    #     all_corresponding_points += [[coord_x + k, coord_y, coord_z]]
                    #     all_corresponding_points += [[coord_x, coord_y + k, coord_z]]
                    #     all_corresponding_points += [[coord_x, coord_y, coord_z +k]]
                    #     all_corresponding_points += [[coord_x + k, coord_y + k, coord_z]]
                    #     all_corresponding_points += [[coord_x + k, coord_y, coord_z + k]]
                    #     all_corresponding_points += [[coord_x, coord_y + k, coord_z + k]]
                    #     all_corresponding_points += [[coord_x + k, coord_y + k, coord_z + k]]
                    #     print("len(all_corresponding_points)", len(all_corresponding_points))
                    #     k += 1
                    
                    # print("all_corresponding_points", x, y, z, all_corresponding_points)
                    # for point in all_corresponding_points:
                    #     position_in_flatten_array = point[2] + point[0] * self.points_per_axis[0] \
                    #                                 + point[1] * self.points_per_axis[0] * self.points_per_axis[1]
                    #     all_corresponding_points2.append(position_in_flatten_array)
                    # print("all_corresponding_points2", x, y, z, all_corresponding_points2)
                    
                    # for position in all_corresponding_points2:
                    #     control_point = control_points[int(position)]
                    #     print("control_point", control_point)

        silent_coarse_momenta_new_scale = dict()
        self.silent_coarse_momenta[scale][key]"""

        """
        residuals_around_points = []
        for (k, point) in enumerate(old_control_points):
            old_spacing = self.deformation_kernel_width
            limits = [(int(max(point[k] - old_spacing/2, 0)), int(min(point[k] + old_spacing/2, residuals.shape[k]))) for k in range(self.dimension)]
            residuals_around_point = residuals[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
            residuals_around_points.append(np.sum(residuals_around_point))
        print("residuals_around_points", residuals_around_points)
        seuil = 0
        self.silent_coarse_momenta = np.ones((len(new_control_points)))
        self.silent_coarse_momenta = []
        for (k, point) in enumerate(old_control_points):
            if residuals_around_points[k] < 1:
                corresponding_new_points = []
                print("\n smooth point", point)
                for (i, point2) in enumerate(new_control_points):
                    if np.linalg.norm(point-point2) <= old_spacing:
                        corresponding_new_points.append(i)
                print("corresponding_new_points", corresponding_new_points)
                #find zones where the points are allowed to differ in haar matrix
                for j in corresponding_new_points:
                    compare = (self.haar_matrix[corresponding_new_points[0]] == self.haar_matrix[j])
                    self.silent_coarse_momenta += list((np.where(compare != True)[0]))
        print("self.silent_coarse_momenta 1", self.silent_coarse_momenta)
        self.silent_coarse_momenta = list(set(self.silent_coarse_momenta)) #keep unique values
        print("self.silent_coarse_momenta", self.silent_coarse_momenta)


        #update COARSE momenta by preserving vector field
        new_coarse_momenta = self.update_vector_field(new_spacing, old_control_points, old_momenta, new_control_points, self.silent_coarse_momenta)
        new_momenta = np.matmul(self.haar_matrix, new_coarse_momenta)

        #save new parameters            
        new_parameters = self.save_new_parameters(new_parameters, new_momenta, new_coarse_momenta, new_control_points)
        print("after adding points", np.shape(self.fixed_effects['control_points']),
        np.shape(self.fixed_effects['momenta']), np.shape(self.fixed_effects['coarse_momenta']))

        print("old_momenta[0]", old_momenta[0])
        print("self.fixed_effects['momenta'][0]", self.fixed_effects['momenta'][0])
        print("self.fixed_effects['coarse_momenta'][0]", self.fixed_effects['coarse_momenta'][0])

        #save new kernel
        self.adapt_kernel(new_kernel_width = new_spacing)  
        """

    def coarse_to_fine2(self, new_parameters, current_iteration, output_dir, dataset, residuals):
        
        print("\n Coarse to fine")

        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, coarse_momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        
        control_points = control_points.cpu().numpy()
        momenta = momenta.cpu().numpy()
        coarse_momenta = coarse_momenta.cpu().numpy()
        
        #control_points = new_parameters["control_points"]
        momenta = new_parameters["momenta"]
        coarse_momenta = new_parameters["coarse_momenta"]
        dimension = momenta.shape[2]

        if self.current_scale > 0:
            self.current_scale -= 1 #unblock smaller scale
            print("self.current_scale", self.current_scale)
            
            #define smooth zones
            print("new_parameters['coarse_momenta'][0][0].J", self.coarser_scale)
            print()
            if self.current_scale < self.coarser_scale:
                print("search smooth zones ...")
                print(new_parameters['coarse_momenta'][0][0].ws)
                wavelet_current_size = [new_parameters['coarse_momenta'][0][0].ws[d][self.current_scale] for d in range(dimension)] #nb of control points                
                #taille des ondelettes en nombre de points controles = des zones à découper
                
                wavelet_size_voxels = [w*residuals.shape[d]/self.points_per_axis[d] for (d,w) in enumerate(wavelet_current_size)]
                print("wavelet_current_size", wavelet_current_size)
                print("wavelet_size_voxels", wavelet_size_voxels)

                #for each wavelet, fetch corresponding control points
                

                
                zones = []
                for i in range(0, self.points_per_axis[0], wavelet_current_size[0]):
                    for j in range(0, self.points_per_axis[1], wavelet_current_size[1]):
                        for k in range(0, self.points_per_axis[2], wavelet_current_size[2]):
                            limit_inf_i = int(math.ceil(i*residuals.shape[0]/self.points_per_axis[0]))
                            limit_inf_j = int(math.ceil(j*residuals.shape[1]/self.points_per_axis[1]))
                            limit_inf_k = int(math.ceil(k*residuals.shape[2]/self.points_per_axis[2]))
                            limit_sup_i = min(limit_inf_i + int(wavelet_size_voxels[0]), residuals.shape[0]) 
                            limit_sup_j = min(limit_inf_j + int(wavelet_size_voxels[1]), residuals.shape[1]) 
                            limit_sup_k = min(limit_inf_k + int(wavelet_size_voxels[2]), residuals.shape[2]) 
                            print("zone", limit_inf_i, limit_inf_j, limit_inf_k)
                            print(limit_sup_i, limit_sup_j, limit_sup_k)
                            zones.append(([i, j, k], np.sum(residuals[limit_inf_i:limit_sup_i, limit_inf_j:limit_sup_j, limit_inf_k:limit_sup_k])))
                
                print("zones", zones)
                print("nb zones", len(zones))
                print(self.points_per_axis)
                print("residuals.shape", residuals.shape)
                print("wavelet_current_size", wavelet_current_size)
                print("new_parameters['coarse_momenta'][0][0].ws", new_parameters['coarse_momenta'][0][0].ws)
                zones_to_silence = [z[0] for z in zones if z[1] < 10]
                self.silent_coarse_momenta[self.current_scale] = zones_to_silence
                print("self.silent_coarse_momenta", self.silent_coarse_momenta)

                #just for check
                for i in range(new_parameters['coarse_momenta'][0][0].wc.shape[0]):
                    for j in range(new_parameters['coarse_momenta'][0][0].wc.shape[1]):
                        for k in range(new_parameters['coarse_momenta'][0][0].wc.shape[2]):
                            print(new_parameters['coarse_momenta'][0][0].haar_coeff_pos_type((i,j,k)))
        
    """
    def add_points_linearly(self, control_points, taux = 0.3):
        self.coarse_to_fine_count += 1
        new_control_points, new_spacing = [], self.original_cp_spacing * np.exp((-1) * taux * self.coarse_to_fine_count)
        #si new_spacing < max_spacing on renvoie un mauvais spacing
        while len(new_control_points) <= len(control_points) and new_spacing > self.maximum_spacing:
            print("try spacing", new_spacing)
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            new_spacing = self.original_cp_spacing * np.exp((-1) * taux * self.coarse_to_fine_count)
            self.coarse_to_fine_count += 1
        
        return new_spacing, new_control_points

    def add_points_regularly(self, current_iteration):
        
        #add points 3 times
        new_spacing = None
        if current_iteration == 1:
            new_spacing = self.max_spacing*3
        if current_iteration == 2 or (new_spacing is not None and self.initial_cp_spacing < new_spacing):
            new_spacing = self.max_spacing*2
        if current_iteration == 3 or (new_spacing is not None and self.initial_cp_spacing < new_spacing):
            new_spacing = self.max_spacing
            
        if new_spacing is not None and self.initial_cp_spacing > new_spacing:
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            
            return new_spacing, new_control_points
        else:
            return self.max_spacing, []

    def add_points_slowly(self, current_iteration, nb_points_origin):
        new_spacing = None
        n = 10 - current_iteration
        if current_iteration > 0 and n > 0:
            new_spacing = self.max_spacing*n
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            while n > 1 and (self.initial_cp_spacing <= new_spacing or len(new_control_points) <= nb_points_origin):
                n = n-1
                new_spacing = self.max_spacing*n
                new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)            
            if n > 0:
                return new_spacing, new_control_points
        
        return self.max_spacing, []

    def add_points_ctf(self, current_spacing, current_iteration, nb_points_origin):
        new_spacing = current_spacing
        new_control_points = []
        N = len(new_control_points)

        while (len(new_control_points) <= nb_points_origin or (2 ** np.floor(np.log(N)/np.log(2)) != N)):
            new_spacing -= 0.1#1
            if new_spacing < self.max_spacing:
                return new_spacing, new_control_points
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                        self.dimension, self.dense_mode)
            N = len(new_control_points)
            print("len(new_control_points)", len(new_control_points))
        
        return new_spacing, new_control_points

    def add_points_only_once(self, current_iteration):
        if current_iteration == 1:
            new_control_points = initialize_control_points(None, self.template, self.max_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            
            return self.max_spacing, new_control_points
        else:
            return self.max_spacing, []
        
    def add_points_same_spacing(self, current_iteration):
        if current_iteration < 15:
            new_control_points = initialize_control_points(None, self.template, self.max_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            
            return self.max_spacing, new_control_points
        else:
            return self.max_spacing, []"""