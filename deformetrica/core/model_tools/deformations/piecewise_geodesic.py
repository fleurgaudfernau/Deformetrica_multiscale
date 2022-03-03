import time
import warnings


from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....in_out.array_readers_and_writers import *
from ....support import utilities

import logging
logger = logging.getLogger(__name__)


def _parallel_transport(*args):

    # read args
    compute_backward, exponential, momenta_to_transport_tR, is_orthogonal = args

    # compute
    if compute_backward:
        return compute_backward, exponential.parallel_transport(momenta_to_transport_tR, is_orthogonal=is_orthogonal)
    else:
        return compute_backward, exponential.parallel_transport(momenta_to_transport_tR, is_orthogonal=is_orthogonal)


class Geodesic:
    """
    Control-point-based LDDMM geodesic.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dense_mode=default.dense_mode,
                 kernel=default.deformation_kernel, shoot_kernel_type=None,
                 tR=default.t0, concentration_of_time_points=default.concentration_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                 nb_components=2, template_tR=None, num_components=None):

        self.concentration_of_time_points = concentration_of_time_points
        self.tmax = None
        self.tmin = None

        self.control_points_tR = None
        self.momenta = None
        self.template_points_tR0 = None
        self.nb_components = int(nb_components)
        self.num_component = num_components

        self.exponential = []
        for i in range(self.nb_components):
            if not (i > 1 and self.num_component[i] == self.num_component[i - 1]):
                self.exponential.append(
                    Exponential(dense_mode=dense_mode, kernel=kernel, shoot_kernel_type=shoot_kernel_type,
                                use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow))

        self.nb_exponentials = self.exponential.__len__()
        self.tR = [None] * (self.nb_exponentials + 1) # tR[0]=tmin, then all the rupture times then tR[-1]=tmax

        # Flags to save extra computations that have already been made in the update methods.
        self.shoot_is_modified = [True]*self.nb_exponentials
        self.shoot_is_modified = True
        self.backward_extension = 0
        self.forward_extension = 0

        # mp.set_sharing_strategy('file_system')
        # self.parallel_transport_pool = mp.Pool(processes=1)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2_for_shoot(self, flag):
        for l in range(self.nb_exponentials):
            self.exponential[l].set_use_rk2_for_shoot(flag)

    def set_use_rk2_for_flow(self, flag):
        for l in range(self.nb_exponentials):
            self.exponential[l].set_use_rk2_for_flow(flag)

    def get_kernel_type(self):
        return self.exponential[0].get_kernel_type()

    def set_kernel(self, kernel):
        for l in range(self.nb_exponentials):
            self.exponential[l].kernel = kernel

    def set_tR(self, tR):
        l = 0
        for i in range(1,self.nb_components):
            if not (i > 1 and self.num_component[i] == self.num_component[i - 1]):
                self.tR[i] = tR[l]
                l += 1
        self.shoot_is_modified = [True]*self.nb_exponentials

    def get_tmin(self):
        return self.tmin

    def set_tmin(self, tmin):
        self.tmin = tmin
        self.tR[0] = tmin
        self.shoot_is_modified[0] = True

    def get_tmax(self):
        return self.tmax

    def set_tmax(self, tmax):
        self.tmax = tmax
        self.tR[-1] = tmax
        self.shoot_is_modified[-1] = True

    def get_template_points_tR(self):
        return self.template_points_tR

    def set_template_points_tR(self, td):
        self.template_points_tR = td
        self.flow_is_modified = True

    def set_control_points_tR(self, cp):
        self.control_points_tR = cp
        self.shoot_is_modified = [True]*self.nb_exponentials

    def set_momenta_tR(self, mom):
        self.momenta = mom
        self.shoot_is_modified = [True]*self.nb_exponentials

    def get_template_points(self, time):
        """
        Returns the position of the landmark points, at the given time.
        Performs a linear interpolation between the two closest available data points.
        """

        assert self.tmin <= time <= self.tmax
        if any(self.shoot_is_modified) or self.flow_is_modified:
            msg = "Asking for deformed template data but the geodesic was modified and not updated"
            warnings.warn(msg)

        times = self.get_times()

        # Deal with the special case of a geodesic reduced to a single point.
        if len(times) == 1:
            logger.info('>> The geodesic seems to be reduced to a single point.')
            return self.template_points_tR

        # Standard case.
        for j in range(1, len(times)):
            if time - times[j] < 0: break

        # j = np.searchsorted(times[:-1], time, side='right')

        # if time <= self.t0:
        #     dt = (self.t0 - self.tmin) / (self.backward_exponential.number_of_time_points - 1)
        #     j = int((time-self.tmin)/dt) + 1
        #
        # else:
        #     dt = (self.tmax - self.t0) / (self.forward_exponential.number_of_time_points - 1)
        #     j = min(len(times)-1,
        #             int((time - self.t0) / dt) + self.backward_exponential.number_of_time_points)
        #
        # assert times[j-1] <= time
        # assert times[j] >= time

        device, _ = utilities.get_best_device(self.exponential[0].kernel.gpu_mode)

        weight_left = utilities.move_data([(times[j] - time) / (times[j] - times[j - 1])], device=device, dtype=self.momenta[0].dtype)
        weight_right = utilities.move_data([(time - times[j - 1]) / (times[j] - times[j - 1])], device=device, dtype=self.momenta[0].dtype)
        template_t = {key: [utilities.move_data(v, device=device) for v in value] for key, value in self.get_template_points_trajectory().items()}

        deformed_points = {key: weight_left * value[j - 1] + weight_right * value[j]
                           for key, value in template_t.items()}

        return deformed_points

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """

        device, _ = utilities.get_best_device(self.exponential[0].kernel.gpu_mode)

        for l in range(self.nb_exponentials):
            length = self.tR[l+1] - self.tR[l]
            self.exponential[l].number_of_time_points = \
                max(1, int(length * self.concentration_of_time_points + 1.5))
            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                if l < 2:
                    self.exponential[l].set_initial_control_points(self.control_points_tR)
                else:
                    self.exponential[l].set_initial_control_points(self.exponential[l-1].control_points_t[-1])

            if self.flow_is_modified:
                if l < 2:
                    self.exponential[l].set_initial_template_points(self.template_points_tR)
                else:
                    self.exponential[l].set_initial_template_points(self.exponential[l - 1].get_template_points(
                            self.exponential[l - 1].momenta_t.__len__() - 1))

            if self.exponential[l].number_of_time_points > 1:
                self.exponential[l].move_data_to_(device=device)
                self.exponential[l].update()

            self.shoot_is_modified[l] = False
        self.flow_is_modified = False


    def get_norm_squared(self,l):
        """
        Get the norm of the geodesic.
        """
        if l < 2: cp = self.control_points_tR
        else: cp = self.exponential[l-1].control_points_t[-1]

        return self.exponential[l].scalar_product(cp, self.momenta[l], self.momenta[l])

    def parallel_transport(self, momenta_to_transport_tR, is_orthogonal=False):
        """
        :param momenta_to_transport_tR: the vector to parallel transport, given at tR and carried at control_points_tR
        :returns: the full trajectory of the parallel transport, from tmin to tmax.
        """
        start = time.perf_counter()

        if any(self.shoot_is_modified):
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)

        transport = []
        for l in range(self.nb_exponentials):
            if self.exponential[l].number_of_time_points > 1:
                transport.append(self.exponential[l].parallel_transport(momenta_to_transport_tR,
                                                                                  is_orthogonal=is_orthogonal))
            else:
                transport.append([momenta_to_transport_tR])
            assert transport[l] is not None

        logger.debug('time taken to compute parallel_transport: ' + str(time.perf_counter() - start))
        transport_concat = transport[0][::-1]
        for l in range(1,self.nb_exponentials):
            transport_concat += transport[l][1:]
        return transport_concat

    ####################################################################################################################
    ### Extension methods:
    ####################################################################################################################

    def get_times(self):
        times = []

        for l in range(self.nb_exponentials):
            times.append([self.tR[l]])
            if self.exponential[l].number_of_time_points > 1:
                times[l] = np.linspace(self.tR[l], self.tR[l+1], num=self.exponential[l].number_of_time_points).tolist()

        times_concat = times[0]
        for l in range(1, self.nb_exponentials):
            times_concat += times[l][1:]
        return times_concat

    def get_control_points_trajectory(self):
        if any(self.shoot_is_modified):
            msg = "Trying to get cp trajectory in a non updated geodesic."
            warnings.warn(msg)

        control_points_t = [self.exponential[0].get_initial_control_points()]
        if self.exponential[0].number_of_time_points > 1:
            control_points_t = self.exponential[0].control_points_t[::-1]

        for l in range(1, self.nb_exponentials):
            if self.exponential[l].number_of_time_points > 1:
                control_points_t += self.exponential[l].control_points_t[1:]

        return control_points_t

    def get_momenta_trajectory(self):
        if any(self.shoot_is_modified):
            msg = "Trying to get mom trajectory in non updated geodesic."
            warnings.warn(msg)

        momenta_t = [self.momenta[0]]
        if self.exponential[0].number_of_time_points > 1:
            length = self.tR[1] - self.tR[0]
            momenta_t = self.exponential[0].momenta_t[::-1]
            momenta_t = [elt / length for elt in momenta_t]

        for l in range(1, self.nb_exponentials):
            if self.exponential[l].number_of_time_points > 1:
                momenta_t += self.exponential[l].momenta_t[1:]

        return momenta_t

    def get_template_points_trajectory(self):
        if any(self.shoot_is_modified) or self.flow_is_modified:
            msg = "Trying to get template trajectory in non updated geodesic."
            warnings.warn(msg)

        template_t = {}
        for key in self.template_points_tR.keys():
            if self.exponential[0].number_of_time_points > 1:
                template_t[key] = self.exponential[0].template_points_t[key][::-1]
            else: template_t[key] = [self.exponential[0].get_initial_template_points()[key]]

            for l in range(1,self.nb_exponentials):
                if self.exponential[l].number_of_time_points > 1:
                    template_t[key] += self.exponential[l].template_points_t[key][1:]

        return template_t

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, root_name, objects_name, objects_extension, template, template_data, output_dir,
              write_adjoint_parameters=False):

        # Core loop ----------------------------------------------------------------------------------------------------
        times = self.get_times()
        for t, time in enumerate(times):
            names = []
            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                name = root_name + '__GeodesicFlow__' + object_name + '__tp_' + str(t) \
                       + ('__age_%.2f' % time) + object_extension
                names.append(name)
            deformed_points = self.get_template_points(time)
            deformed_data = template.get_deformed_data(deformed_points, template_data)
            template.write(output_dir, names,
                           {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

        # Optional writing of the control points and momenta -----------------------------------------------------------
        if write_adjoint_parameters:
            control_points_t = [elt.detach().cpu().numpy() for elt in self.get_control_points_trajectory()]
            momenta_t = [elt.detach().cpu().numpy() for elt in self.get_momenta_trajectory()]
            for t, (time, control_points, momenta) in enumerate(zip(times, control_points_t, momenta_t)):
                write_2D_array(control_points, output_dir, root_name + '__GeodesicFlow__ControlPoints__tp_' + str(t)
                               + ('__age_%.2f' % time) + '.txt')
                write_2D_array(momenta, output_dir, root_name + '__GeodesicFlow__Momenta__tp_' + str(t)
                               + ('__age_%.2f' % time) + '.txt')
