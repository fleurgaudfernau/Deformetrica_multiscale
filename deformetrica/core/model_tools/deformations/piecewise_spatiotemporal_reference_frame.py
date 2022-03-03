import torch

from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....core.model_tools.deformations.piecewise_geodesic import Geodesic
from ....in_out.array_readers_and_writers import *


class SpatiotemporalReferenceFrame:
    """
    Control-point-based LDDMM spatio-temporal reference frame, based on exp-parallelization.
    See "Learning distributions of shape trajectories from longitudinal datasets: a hierarchical model on a manifold
    of diffeomorphisms", Bône et al. (2018), in review.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dense_mode=default.dense_mode,
                 kernel=default.deformation_kernel, shoot_kernel_type=default.shoot_kernel_type, tR=default.t0,
                 concentration_of_time_points=default.concentration_of_time_points,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                 template_tR=None, nb_components=2, num_components=None):

        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel, shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points, use_rk2_for_shoot=use_rk2_for_shoot,
            use_rk2_for_flow=use_rk2_for_flow)

        self.geodesic = Geodesic(
            dense_mode=dense_mode, kernel=kernel, tR=tR, concentration_of_time_points=concentration_of_time_points,
            use_rk2_for_shoot=True, use_rk2_for_flow=use_rk2_for_flow, template_tR=template_tR,
            nb_components=nb_components, num_components=num_components)

        self.modulation_matrix_tR = None
        self.projected_modulation_matrix_tR = None
        self.projected_modulation_matrix_t = None
        self.number_of_sources = None

        self.transport_is_modified = True

        self.times = None
        self.template_points_t = None
        self.control_points_t = None

    def clone(self):
        raise NotImplementedError  # TODO
        # clone = SpatiotemporalReferenceFrame()
        #
        # clone.geodesic = self.geodesic.clone()
        # clone.exponential = self.exponential.clone()
        #
        # if self.modulation_matrix_t0 is not None:
        #     clone.modulation_matrix_t0 = self.modulation_matrix_t0.clone()
        # if self.projected_modulation_matrix_t0 is not None:
        #     clone.projected_modulation_matrix_t0 = self.projected_modulation_matrix_t0.clone()
        # if self.projected_modulation_matrix_t is not None:
        #     clone.projected_modulation_matrix_t = [elt.clone() for elt in self.projected_modulation_matrix_t]
        # clone.number_of_sources = self.number_of_sources
        #
        # clone.transport_is_modified = self.transport_is_modified
        # clone.backward_extension = self.backward_extension
        # clone.forward_extension = self.forward_extension
        #
        # clone.times = self.times
        # if self.template_points_t is not None:
        #     clone.template_points_t = {key: [elt.clone() for elt in value]
        #                                for key, value in self.template_points_t.item()}
        # if self.control_points_t is not None:
        #     clone.control_points_t = [elt.clone() for elt in self.control_points_t]

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2_for_shoot(self, flag):  # Cannot modify the shoot integration of the geodesic, which require rk2.
        self.exponential.set_use_rk2_for_shoot(flag)

    def set_use_rk2_for_flow(self, flag):
        self.exponential.set_use_rk2_for_shoot(flag)
        self.geodesic.set_use_rk2_for_flow(flag)

    def set_kernel(self, kernel):
        self.geodesic.set_kernel(kernel)
        self.exponential.set_kernel(kernel)

    def get_kernel_type(self):
        return self.exponential.kernel.kernel_type

    def get_kernel_width(self):
        return self.exponential.kernel.kernel_width

    def get_concentration_of_time_points(self):
        return self.geodesic.concentration_of_time_points

    def set_concentration_of_time_points(self, ctp):
        self.geodesic.concentration_of_time_points = ctp

    def set_number_of_time_points(self, ntp):
        self.exponential.number_of_time_points = ntp

    def set_template_points_tR(self, td):
        self.geodesic.set_template_points_tR(td)

    def set_control_points_tR(self, cp):
        self.geodesic.set_control_points_tR(cp)
        self.transport_is_modified = True

    def set_momenta(self, mom):
        self.geodesic.set_momenta_tR(mom)
        self.transport_is_modified = True

    def set_modulation_matrix_tR(self, mm):
        self.modulation_matrix_tR = mm
        self.number_of_sources = mm.size()[1]
        self.transport_is_modified = True

    def set_tR(self, tR):
        self.geodesic.set_tR(tR)
        self.transport_is_modified = True

    def get_tmin(self):
        return self.geodesic.get_tmin()

    def set_tmin(self, tmin, optimize=False):
        self.geodesic.set_tmin(tmin)
        self.transport_is_modified = True

    def get_tmax(self):
        return self.geodesic.get_tmax()

    def set_tmax(self, tmax, optimize=False):
        self.geodesic.set_tmax(tmax)
        self.transport_is_modified = True

    def get_template_points_exponential_parameters(self, time, sources):

        # Assert for coherent length of attribute lists.
        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.control_points_t) == len(
            self.projected_modulation_matrix_t) == len(self.times)

        # Deal with the special case of a geodesic reduced to a single point.
        if len(self.times) == 1:
            logger.info('>> The spatiotemporal reference frame geodesic seems to be reduced to a single point.')
            initial_template_points = {key: value[0] for key, value in self.template_points_t.items()}
            initial_control_points = self.control_points_t[0]
            initial_momenta = torch.mm(self.projected_modulation_matrix_t[0], sources.unsqueeze(1)).view(
                self.geodesic.momenta[0].size())

        # Standard case.
        else:
            index, weight_left, weight_right = self._get_interpolation_index_and_weights(time)
            template_points = {key: weight_left * value[index - 1] + weight_right * value[index]
                               for key, value in self.template_points_t.items()}
            control_points = weight_left * self.control_points_t[index - 1] + weight_right * self.control_points_t[index]
            modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] + weight_right * self.projected_modulation_matrix_t[index]
            space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.momenta[0].size())

            initial_template_points = template_points
            initial_control_points = control_points
            initial_momenta = space_shift

        return initial_template_points, initial_control_points, initial_momenta

    def get_template_points(self, time, sources, device=None):

        # Assert for coherent length of attribute lists.
        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.control_points_t) \
               == len(self.projected_modulation_matrix_t) == len(self.times)

        # Deal with the special case of a geodesic reduced to a single point.
        if len(self.times) == 1:
            logger.info('>> The spatiotemporal reference frame geodesic seems to be reduced to a single point.')
            self.exponential.set_initial_template_points({key: value[0]
                                                          for key, value in self.template_points_t.items()})
            self.exponential.set_initial_control_points(self.control_points_t[0])
            self.exponential.set_initial_momenta(torch.mm(self.projected_modulation_matrix_t[0],
                                                          sources.unsqueeze(1)).view(self.geodesic.momenta[0].size()))

        # Standard case.
        else:
            index, weight_left, weight_right = self._get_interpolation_index_and_weights(time)
            template_points = {key: weight_left * value[index - 1] + weight_right * value[index]
                               for key, value in self.template_points_t.items()}
            control_points = weight_left * self.control_points_t[index - 1] + weight_right * self.control_points_t[index]
            modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] \
                                + weight_right * self.projected_modulation_matrix_t[index]
            space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.momenta[0].size())

            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_control_points(control_points)
            self.exponential.set_initial_momenta(space_shift)

        if device is not None:
            self.exponential.move_data_to_(device)
        self.exponential.update()
        return self.exponential.get_template_points()

    def _get_interpolation_index_and_weights(self, time):
        for index in range(1, len(self.times)):
            if time.data.cpu().numpy() - self.times[index] < 0:
                break
        weight_left = (self.times[index] - time) / (self.times[index] - self.times[index - 1])
        weight_right = (time - self.times[index - 1]) / (self.times[index] - self.times[index - 1])
        return index, weight_left, weight_right

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Update the geodesic, and compute the parallel transport of each column of the modulation matrix along
        this geodesic, ignoring the tangential components.
        """

        device = self.geodesic.control_points_tR.device

        # Update the geodesic.
        self.geodesic.update()

        # Convenient attributes for later use.
        self.times = self.geodesic.get_times()
        self.template_points_t = self.geodesic.get_template_points_trajectory()
        self.control_points_t = self.geodesic.get_control_points_trajectory()

        if self.transport_is_modified:
            # Projects the modulation_matrix_t0 attribute columns.
            self._update_projected_modulation_matrix_tR(device=device)

            # Initializes the projected_modulation_matrix_t attribute size.
            self.projected_modulation_matrix_t = \
                [torch.zeros(self.projected_modulation_matrix_tR.size(), dtype=self.modulation_matrix_tR.dtype, device=device)
                 for _ in range(len(self.control_points_t))]

            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                space_shift_tR = self.projected_modulation_matrix_tR[:, s].contiguous().view(self.geodesic.momenta[0].size())
                space_shift_t = self.geodesic.parallel_transport(space_shift_tR, is_orthogonal=True)

                # Set the result correctly in the projected_modulation_matrix_t attribute.
                for t, space_shift in enumerate(space_shift_t):
                    self.projected_modulation_matrix_t[t][:, s] = space_shift.view(-1)

            self.transport_is_modified = False

        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.control_points_t) \
               == len(self.times) == len(self.projected_modulation_matrix_t), \
            "That's weird: len(self.template_points_t[list(self.template_points_t.keys())[0]]) = %d, " \
            "len(self.control_points_t) = %d, len(self.times) = %d,  len(self.projected_modulation_matrix_t) = %d" % \
            (len(self.template_points_t[list(self.template_points_t.keys())[0]]), len(self.control_points_t),
             len(self.times), len(self.projected_modulation_matrix_t))

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    def _update_projected_modulation_matrix_tR(self, device='cpu'):
        self.projected_modulation_matrix_tR = self.modulation_matrix_tR.clone()
        momenta_orth = self.ortho(self.geodesic.momenta)

        for l in range(self.geodesic.exponential.__len__()):
            norm_squared = self.geodesic.exponential[l].scalar_product(
                self.geodesic.exponential[l].get_initial_control_points(), momenta_orth[l], momenta_orth[l])

            if norm_squared != 0:
                for s in range(self.number_of_sources):
                    space_shift_tR = self.projected_modulation_matrix_tR[:, s].contiguous().view(self.geodesic.momenta[0].size()).clone()
                    sp = self.geodesic.exponential[l].scalar_product(
                        self.geodesic.exponential[l].get_initial_control_points(), momenta_orth[l], space_shift_tR) / norm_squared

                    projected_space_shift_tR = space_shift_tR - sp * momenta_orth[l]
                    self.projected_modulation_matrix_tR[:, s] = projected_space_shift_tR.view(-1).contiguous()

    def ortho(self, momenta):
        momenta_ortho = momenta.copy()
        for k in range(1,momenta.__len__()):
            for l in range(k):
                norm_squared = self.geodesic.exponential[l].scalar_product(self.geodesic.exponential[l].get_initial_control_points(),
                                                                               self.geodesic.momenta[l],
                                                                               self.geodesic.momenta[l])
                if norm_squared != 0:
                    sp_to_ortho = self.geodesic.exponential[l].scalar_product(self.geodesic.exponential[l].get_initial_control_points(),
                                                                               self.geodesic.momenta[k],
                                                                               momenta_ortho[l]) / norm_squared
                    momenta_ortho[k] = momenta_ortho[k] - sp_to_ortho*momenta_ortho[l]
        return momenta_ortho

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, root_name, objects_name, objects_extension, template, template_data, output_dir,
              write_adjoint_parameters=False, write_exponential_flow=False):

        # Write the geodesic -------------------------------------------------------------------------------------------
        self.geodesic.write(root_name, objects_name, objects_extension, template, template_data, output_dir,
                            write_adjoint_parameters)

        # Write the orthogonal flow ------------------------------------------------------------------------------------
        # Plot the flow up to three standard deviations.
        self.exponential.number_of_time_points = 1 + 3 * (self.exponential.number_of_time_points - 1)
        for s in range(self.number_of_sources):

            # Direct flow.
            space_shift = self.projected_modulation_matrix_tR[:, s].contiguous().view(
                self.geodesic.momenta[0].size())
            self.exponential.set_initial_template_points(self.geodesic.template_points_tR)
            self.exponential.set_initial_control_points(self.geodesic.control_points_tR)
            self.exponential.set_initial_momenta(space_shift)
            self.exponential.update()

            for j in range(self.exponential.number_of_time_points):
                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__GeometricMode_' + str(s) + '__' + object_name \
                           + '__' + str(self.exponential.number_of_time_points - 1 + j) \
                           + ('__+%.2f_sigma' % (3. * float(j) / (self.exponential.number_of_time_points - 1))) \
                           + object_extension
                    names.append(name)
                deformed_points = self.exponential.get_template_points(j)
                deformed_data = template.get_deformed_data(deformed_points, template_data)
                template.write(output_dir, names,
                               {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

            # Indirect flow.
            space_shift = self.projected_modulation_matrix_tR[:, s].contiguous().view(
                self.geodesic.momenta[0].size())
            self.exponential.set_initial_template_points(self.geodesic.template_points_tR)
            self.exponential.set_initial_control_points(self.geodesic.control_points_tR)
            self.exponential.set_initial_momenta(- space_shift)
            self.exponential.update()

            for j in range(self.exponential.number_of_time_points):
                if j == 0:
                    continue
                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__GeometricMode_' + str(s) + '__' + object_name \
                           + '__' + str(self.exponential.number_of_time_points - 1 - j) \
                           + ('__-%.2f_sigma' % (3. * float(j) / (self.exponential.number_of_time_points - 1))) \
                           + object_extension
                    names.append(name)
                deformed_points = self.exponential.get_template_points(j)
                deformed_data = template.get_deformed_data(deformed_points, template_data)
                template.write(output_dir, names,
                               {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

        # Correctly resets the initial number of time points.
        self.exponential.number_of_time_points = 1 + (self.exponential.number_of_time_points - 1) // 3

        # Optionally write the projected modulation matrices along the geodesic flow -----------------------------------
        if write_adjoint_parameters:
            times = self.geodesic.get_times()
            for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_modulation_matrix_t)):
                write_2D_array(
                    modulation_matrix.detach().cpu().numpy(),
                    output_dir,
                    root_name + '__GeodesicFlow__ModulationMatrix__tp_' + str(t) + ('__age_%.2f' % time) + '.txt')

        # Optionally write the exp-parallel curves and associated flows (massive writing) ------------------------------
        if write_exponential_flow:
            times = self.geodesic.get_times()
            for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_modulation_matrix_t)):
                for s in range(self.number_of_sources):

                    # Forward.
                    space_shift = modulation_matrix[:, s].contiguous().view(self.geodesic.momenta[0].size())
                    self.exponential.set_initial_template_points({key: value[t]
                                                                  for key, value in self.template_points_t.items()})
                    self.exponential.set_initial_control_points(self.control_points_t[t])
                    self.exponential.set_initial_momenta(space_shift)
                    self.exponential.update()

                    names = []
                    for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                        name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
                               + ('__age_%.2f' % time) + '__ForwardExponentialFlow'
                        names.append(name)
                    self.exponential.write_flow(names, objects_extension, template, template_data, output_dir,
                                                write_adjoint_parameters)

                    # Backward
                    self.exponential.set_initial_momenta(- space_shift)
                    self.exponential.update()

                    names = []
                    for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                        name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
                               + ('__age_%.2f' % time) + '__BackwardExponentialFlow'
                        names.append(name)
                    self.exponential.write_flow(names, objects_extension, template, template_data, output_dir,
                                                write_adjoint_parameters)
