import unittest
import numpy as np
import torch

import deformetrica as dfca

from torch.autograd import Variable


class ShootingTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_geodesic_shooting(self):
        """
        Test the shooting with a single cp. tests with (tmin=t0=0,tmax=1 ; tmin=-1,tmax=t0=0.; tmin=-1,t0=0,tmax=1)
        """
        control_points = np.array([[0.1, 0.2, 0.2]])
        momenta = np.array([[1., 0.2, 0.]])

        control_points_torch = Variable(torch.from_numpy(control_points).type(dfca.default.tensor_scalar_type))
        momenta_torch = Variable(torch.from_numpy(momenta).type(dfca.default.tensor_scalar_type))

        geodesic = dfca.deformations.Geodesic(
            dense_mode=False,
            kernel=dfca.kernels.factory('torch', kernel_width=0.01),
            t0=0.,
            use_rk2_for_shoot=True,
            use_rk2_for_flow=True,
            concentration_of_time_points=10
        )

        geodesic.set_momenta_t0(momenta_torch)
        geodesic.set_control_points_t0(control_points_torch)
        geodesic.set_tmin(-1.)
        geodesic.set_tmax(0.)
        geodesic.update()

        cp_traj = geodesic.get_control_points_trajectory()
        mom_traj = geodesic.get_momenta_trajectory()
        times_traj = geodesic.get_times()

        self.assertTrue(len(cp_traj) == len(mom_traj))
        self.assertTrue(len(times_traj) == len(cp_traj))

        for (cp, mom, time) in zip(cp_traj, mom_traj, times_traj):
            self.assertTrue(np.allclose(cp.detach().numpy(), control_points + time * momenta))
            self.assertTrue(np.allclose(mom.detach().numpy(), momenta))

        geodesic.set_tmin(-1.)
        geodesic.set_tmax(0.)
        geodesic.set_t0(0.)
        geodesic.update()

        cp_traj = geodesic.get_control_points_trajectory()
        mom_traj = geodesic.get_momenta_trajectory()
        times_traj = geodesic.get_times()

        self.assertTrue(len(cp_traj) == len(mom_traj))
        self.assertTrue(len(times_traj) == len(cp_traj))

        for (cp, mom, time) in zip(cp_traj, mom_traj, times_traj):
            # logger.info(time, cp.detach().numpy(), control_points + time * momenta)
            self.assertTrue(np.allclose(cp.detach().numpy(), control_points + time * momenta))
            self.assertTrue(np.allclose(mom.detach().numpy(), momenta))

        geodesic.set_tmin(-1.)
        geodesic.set_tmax(0.)
        geodesic.set_t0(0.)
        geodesic.update()

        cp_traj = geodesic.get_control_points_trajectory()
        mom_traj = geodesic.get_momenta_trajectory()
        times_traj = geodesic.get_times()

        self.assertTrue(len(cp_traj) == len(mom_traj))
        self.assertTrue(len(times_traj) == len(cp_traj))

        for (cp, mom, time) in zip(cp_traj, mom_traj, times_traj):
            self.assertTrue(np.allclose(cp.detach().numpy(), control_points + time * momenta))
            self.assertTrue(np.allclose(mom.detach().numpy(), momenta))
