import logging
logger = logging.getLogger(__name__)

import deformetrica as dfca
import pykeops

import pickle
import unittest
import torch
import numpy as np


class KernelFactoryTest(unittest.TestCase):
    def test_instantiate_abstract_class(self):
        with self.assertRaises(TypeError):
            dfca.kernels.AbstractKernel()

    def test_unknown_kernel_string(self):
        with self.assertRaises(TypeError):
            dfca.kernels.factory('unknown_type')

    def test_non_cuda_kernel_factory(self):
        for k in [dfca.kernels.Type.TORCH, dfca.kernels.Type.KEOPS]:
            logging.debug("testing kernel=%s" % k)
            instance = dfca.kernels.factory(k, kernel_width=1.)
            self.__isKernelValid(instance)

    def test_no_kernel_type_from_string(self):
        for k in ['no_kernel', 'no-kernel', 'no kernel', 'undefined', 'UNDEFINED']:
            logging.debug("testing kernel= %s" % k)
            instance = dfca.kernels.factory(k, kernel_width=1.)
            self.assertIsNone(instance)

    def test_non_cuda_kernel_factory_from_string(self):
        for k in ['torch', 'TORCH', 'keops', 'KEOPS']:
            logging.debug("testing kernel= %s" % k)
            instance = dfca.kernels.factory(k, kernel_width=1.)
            self.__isKernelValid(instance)

    def __isKernelValid(self, instance):
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, dfca.kernels.AbstractKernel)
        self.assertEqual(instance.kernel_width, 1.)


class KernelTestBase(unittest.TestCase):
    def setUp(self):
        dfca.default.update_dtype('float64')
        self.torch_dtype = dfca.utils.get_torch_dtype(dfca.default.dtype)

        torch.manual_seed(42)  # for reproducibility
        torch.set_printoptions(precision=30)  # for more precision when printing tensor

        self.x = torch.rand((4, 3), dtype=self.torch_dtype)
        self.y = torch.rand((4, 3), dtype=self.torch_dtype)
        self.p = torch.rand((4, 3), dtype=self.torch_dtype)
        self.expected_convolve_res = torch.tensor([
            [0.442066994263886070548608131503, 0.674582639132687567062873768009, 0.823952763352665096263649502362],
            [0.453613208431209002924333617557, 0.696584221587193019864514553774, 1.024381845902817111948479578132],
            [0.442705483227435858673004531738, 0.650869945838532748538796113280, 1.399729801162817866000409594562],
            [0.514609773097077893844186746719, 0.798936940519222593692916234431, 1.009915759134895285598076952738]], dtype=self.torch_dtype)

        self.expected_convolve_gradient_res = torch.tensor([
            [ 0.055114156220794276175301007470,  0.119300787807738090107179118604, 0.075544184568281505520737084680],
            [ 0.374664655710255534160069146310, -0.082597015881930607728023119307, -0.241403157091644060550095218787],
            [-0.512730247054557164432253557607, -0.316429061213168882904511747256, 0.194564631495594497767598340943],
            [ 0.082951435123507416546928538992,  0.279725289287361400525355747959, -0.028705658972231928860452399022]], dtype=self.torch_dtype)

        super().setUp()

    def _assert_tensor_close(self, t1, t2, precision=1e-15):
        if t1.requires_grad is True:
            t1 = t1.detach()
        if t2.requires_grad is True:
            t2 = t2.detach()

        # logger.info(t1)
        # logger.info(t2)
        # logger.info(t1 - t2)
        self.assertTrue(np.allclose(t1, t2, rtol=precision, atol=precision),
                        'Tested tensors are not within acceptable tolerance levels')

    def _assert_same_kernels(self, k1, k2):
        self.assertTrue(type(k1), type(k2))
        self.assertEqual(k1.kernel_type, k2.kernel_type)
        self.assertEqual(k1.kernel_width, k2.kernel_width)

    def test_multi_instance(self):
        k1 = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=1.)
        k2 = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=1.)
        k3 = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=1.1)
        k4 = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=1.)
        k5 = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=1.)
        print("id(k1)=" + str(id(k1)) + ", id(k2)=" + str(id(k2)))
        print("hash(k1)=" + str(hash(k1)) + ", hash(k2)=" + str(hash(k2)))

        # check value equality
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(k4, k5)
        self.assertNotEqual(k1, k4)

        # check identity equality
        self.assertEqual(id(k1), id(k2))
        self.assertNotEqual(id(k1), id(k3))
        self.assertEqual(id(k4), id(k5))
        self.assertNotEqual(id(k1), id(k4))


class TorchKernelTest(KernelTestBase):
    def setUp(self):
        super().setUp()

    def test_convolve_cpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=1.)
        res = kernel_instance.convolve(self.x, self.y, self.p)
        self._assert_tensor_close(res, self.expected_convolve_res)

    def test_convolve_gradient_cpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=1.)
        res = kernel_instance.convolve_gradient(self.x, self.x)
        self._assert_tensor_close(res, self.expected_convolve_gradient_res)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_convolve_gpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.TORCH, gpu_mode=dfca.GpuMode.FULL, kernel_width=1.)

        device = torch.device('cuda:0')
        x_gpu = dfca.utils.move_data(self.x, device=device)
        y_gpu = dfca.utils.move_data(self.y, device=device)
        p_gpu = dfca.utils.move_data(self.p, device=device)

        res = kernel_instance.convolve(x_gpu, y_gpu, p_gpu)
        self.assertEqual(device, res.device)
        self._assert_tensor_close(res.cpu(), self.expected_convolve_res)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_convolve_gradient_gpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.TORCH, gpu_mode=dfca.GpuMode.FULL, kernel_width=1.)

        device = torch.device('cuda:0')
        x_gpu = dfca.utils.move_data(self.x, device=device)

        res = kernel_instance.convolve_gradient(x_gpu, x_gpu)
        self.assertEqual(device, res.device)
        self._assert_tensor_close(res.cpu(), self.expected_convolve_gradient_res)

    def test_pickle(self):
        logger.info('torch.__version__=' + torch.__version__)

        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=1.)

        # serialize/pickle
        serialized_kernel = pickle.dumps(kernel_instance)
        # deserialize/unpickle
        deserialized_kernel = pickle.loads(serialized_kernel)

        self._assert_same_kernels(kernel_instance, deserialized_kernel)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_gpu_mode(self):

        # AUTO = auto(),
        # FULL = auto(),
        # NONE = auto(),
        # KERNEL = auto()

        for gpu_mode in dfca.GpuMode:
            print(gpu_mode.name)
            if gpu_mode is dfca.GpuMode.AUTO:
                continue    # TODO

            kernel_instance = dfca.kernels.factory(dfca.kernels.Type.TORCH, gpu_mode=gpu_mode, kernel_width=1.)
            res = kernel_instance.convolve(self.x, self.y, self.p)

            if gpu_mode is dfca.GpuMode.FULL:
                self.assertEqual('cuda', res.device.type)
                res = res.cpu()

            self.assertEqual('cpu', res.device.type)
            self._assert_tensor_close(res, self.expected_convolve_res)


class KeopsKernelTest(KernelTestBase):
    def setUp(self):
        super().setUp()

    def test_convolve_cpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=1.)
        res = kernel_instance.convolve(self.x, self.y, self.p)
        self._assert_tensor_close(res, self.expected_convolve_res)

    def test_convolve_gradient_cpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=1.)
        res = kernel_instance.convolve_gradient(self.x, self.x)
        self._assert_tensor_close(res, self.expected_convolve_gradient_res)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_convolve_gpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.KEOPS, gpu_mode=dfca.GpuMode.FULL, kernel_width=1.)

        device = torch.device('cuda:0')
        x_gpu = dfca.utils.move_data(self.x, device=device)
        y_gpu = dfca.utils.move_data(self.y, device=device)
        p_gpu = dfca.utils.move_data(self.p, device=device)

        res = kernel_instance.convolve(x_gpu, y_gpu, p_gpu)
        self.assertEqual(device, res.device)
        self._assert_tensor_close(res.cpu(), self.expected_convolve_res)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_convolve_gradient_gpu(self):
        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.KEOPS, gpu_mode=dfca.GpuMode.FULL, kernel_width=1.)

        device = torch.device('cuda:0')
        x_gpu = dfca.utils.move_data(self.x, device=device)

        res = kernel_instance.convolve_gradient(x_gpu, x_gpu)
        self.assertEqual(device, res.device)
        self._assert_tensor_close(res.cpu(), self.expected_convolve_gradient_res)

    def test_pickle(self):
        logger.info('torch.__version__=' + torch.__version__)
        logger.info('pykeops.__version__=' + pykeops.__version__)

        kernel_instance = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=1.)

        # serialize/pickle
        serialized_kernel = pickle.dumps(kernel_instance)
        # deserialize/unpickle
        deserialized_kernel = pickle.loads(serialized_kernel)

        self._assert_same_kernels(kernel_instance, deserialized_kernel)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_gpu_mode(self):

        for gpu_mode, cuda_type in [(gpu_mode, cuda_type)
                                    for gpu_mode in [gpu_mode for gpu_mode in dfca.GpuMode]
                                    for cuda_type in ['float32', 'float64']]:
            if gpu_mode is dfca.GpuMode.AUTO:
                continue   # TODO
            print('gpu_mode: ' + str(gpu_mode) + ', cuda_type: ' + cuda_type)

            kernel_instance = dfca.kernels.factory(dfca.kernels.Type.KEOPS, gpu_mode=gpu_mode, kernel_width=1., cuda_type=cuda_type)

            x = self.x
            y = self.y
            p = self.p

            if cuda_type == 'float32':
                dfca.default.update_dtype('float32')
                x = self.x.float()
                y = self.y.float()
                p = self.p.float()

            res = kernel_instance.convolve(x, y, p)

            if gpu_mode is dfca.GpuMode.FULL:
                self.assertEqual('cuda', res.device.type)
                res = res.cpu()

            self.assertEqual('cpu', res.device.type)
            self._assert_tensor_close(res, self.expected_convolve_res, precision=1e-7)


class KeopsVersusCuda(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dfca.default.update_dtype('float64')
        self.torch_dtype = dfca.utils.get_torch_dtype(dfca.default.dtype)
        self.tensor_scalar_type = dfca.default.tensor_scalar_type
        self.precision = 1e-12

    def test_keops_and_torch_gaussian_convolve_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        # tensor_scalar_type = torch.cuda.FloatTensor
        # tensor_scalar_type = torch.FloatTensor

        # Instantiate the needed objects.
        keops_kernel = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=kernel_width)
        torch_kernel = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=kernel_width)
        random_control_points_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_control_points_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_momenta_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_momenta_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()

        # Compute the desired forward quantities.
        keops_convolve_11 = keops_kernel.convolve(random_control_points_1, random_control_points_1, random_momenta_1)
        torch_convolve_11 = torch_kernel.convolve(random_control_points_1, random_control_points_1, random_momenta_1)
        keops_convolve_12 = keops_kernel.convolve(random_control_points_1, random_control_points_2, random_momenta_2)
        torch_convolve_12 = torch_kernel.convolve(random_control_points_1, random_control_points_2, random_momenta_2)

        # Compute the desired backward quantities.
        keops_total_12 = torch.dot(random_momenta_1.view(-1), keops_convolve_12.view(-1))
        torch_total_12 = torch.dot(random_momenta_1.view(-1), torch_convolve_12.view(-1))

        [keops_dcp_1, keops_dcp_2, keops_dmom_1, keops_dmom_2] = torch.autograd.grad(
            keops_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])
        [torch_dcp_1, torch_dcp_2, torch_dmom_1, torch_dmom_2] = torch.autograd.grad(
            torch_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])

        # Convert back to numpy.
        keops_convolve_11 = keops_convolve_11.detach().cpu().numpy()
        torch_convolve_11 = torch_convolve_11.detach().cpu().numpy()
        keops_convolve_12 = keops_convolve_12.detach().cpu().numpy()
        torch_convolve_12 = torch_convolve_12.detach().cpu().numpy()
        keops_dcp_1 = keops_dcp_1.detach().cpu().numpy()
        keops_dcp_2 = keops_dcp_2.detach().cpu().numpy()
        keops_dmom_1 = keops_dmom_1.detach().cpu().numpy()
        keops_dmom_2 = keops_dmom_2.detach().cpu().numpy()
        torch_dcp_1 = torch_dcp_1.detach().cpu().numpy()
        torch_dcp_2 = torch_dcp_2.detach().cpu().numpy()
        torch_dmom_1 = torch_dmom_1.detach().cpu().numpy()
        torch_dmom_2 = torch_dmom_2.detach().cpu().numpy()

        # Check for equality.
        self.assertTrue(np.allclose(keops_convolve_11, torch_convolve_11, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_convolve_12, torch_convolve_12, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dcp_1, torch_dcp_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dcp_2, torch_dcp_2, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dmom_1, torch_dmom_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dmom_2, torch_dmom_2, rtol=self.precision, atol=self.precision))

    def test_keops_and_torch_varifold_convolve_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        # tensor_scalar_type = torch.cuda.FloatTensor

        # Instantiate the needed objects.
        keops_kernel = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=kernel_width)
        torch_kernel = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=kernel_width)
        random_points_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_points_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_normals_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_normals_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_areas_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, 1)).type(self.tensor_scalar_type).requires_grad_()
        random_areas_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, 1)).type(self.tensor_scalar_type).requires_grad_()

        # Compute the desired forward quantities.
        keops_convolve_11 = keops_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_1, random_normals_1), random_areas_1, mode='varifold')
        torch_convolve_11 = torch_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_1, random_normals_1), random_areas_1, mode='varifold')
        keops_convolve_12 = keops_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_2, random_normals_2), random_areas_2, mode='varifold')
        torch_convolve_12 = torch_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_2, random_normals_2), random_areas_2, mode='varifold')

        # Compute the desired backward quantities.
        keops_total_12 = torch.dot(random_areas_1.view(-1), keops_convolve_12.view(-1))
        torch_total_12 = torch.dot(random_areas_1.view(-1), torch_convolve_12.view(-1))

        [keops_dp_1, keops_dp_2, keops_dn_1, keops_dn_2, keops_da_1, keops_da_2] = torch.autograd.grad(
            keops_total_12,
            [random_points_1, random_points_2, random_normals_1, random_normals_2, random_areas_1, random_areas_2])
        [torch_dp_1, torch_dp_2, torch_dn_1, torch_dn_2, torch_da_1, torch_da_2] = torch.autograd.grad(
            torch_total_12,
            [random_points_1, random_points_2, random_normals_1, random_normals_2, random_areas_1, random_areas_2])

        # Convert back to numpy.
        keops_convolve_11 = keops_convolve_11.detach().cpu().numpy()
        torch_convolve_11 = torch_convolve_11.detach().cpu().numpy()
        keops_convolve_12 = keops_convolve_12.detach().cpu().numpy()
        torch_convolve_12 = torch_convolve_12.detach().cpu().numpy()
        keops_dp_1 = keops_dp_1.detach().cpu().numpy()
        keops_dp_2 = keops_dp_2.detach().cpu().numpy()
        keops_dn_1 = keops_dn_1.detach().cpu().numpy()
        keops_dn_2 = keops_dn_2.detach().cpu().numpy()
        keops_da_1 = keops_da_1.detach().cpu().numpy()
        keops_da_2 = keops_da_2.detach().cpu().numpy()
        torch_dp_1 = torch_dp_1.detach().cpu().numpy()
        torch_dp_2 = torch_dp_2.detach().cpu().numpy()
        torch_dn_1 = torch_dn_1.detach().cpu().numpy()
        torch_dn_2 = torch_dn_2.detach().cpu().numpy()
        torch_da_1 = torch_da_1.detach().cpu().numpy()
        torch_da_2 = torch_da_2.detach().cpu().numpy()

        # Check for equality.
        self.assertTrue(np.allclose(keops_convolve_11, torch_convolve_11, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_convolve_12, torch_convolve_12, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dp_1, torch_dp_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dp_2, torch_dp_2, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dn_1, torch_dn_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dn_2, torch_dn_2, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_da_1, torch_da_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_da_2, torch_da_2, rtol=self.precision, atol=self.precision))

    def test_keops_and_torch_convolve_gradient_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3


        # Instantiate the needed objects.
        keops_kernel = dfca.kernels.factory(dfca.kernels.Type.KEOPS, kernel_width=kernel_width)
        torch_kernel = dfca.kernels.factory(dfca.kernels.Type.TORCH, kernel_width=kernel_width)
        random_control_points_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_control_points_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_momenta_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()
        random_momenta_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(self.tensor_scalar_type).requires_grad_()

        # Compute the desired forward quantities.
        keops_convolve_gradient_11 = keops_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1)
        torch_convolve_gradient_11 = torch_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1)
        keops_convolve_gradient_11_bis = keops_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_1, random_momenta_1)
        torch_convolve_gradient_11_bis = torch_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_1, random_momenta_1)
        keops_convolve_gradient_12 = keops_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_2, random_momenta_2)
        torch_convolve_gradient_12 = torch_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_2, random_momenta_2)

        # Compute the desired backward quantities.
        keops_total_12 = torch.dot(random_momenta_1.view(-1), keops_convolve_gradient_12.contiguous().view(-1))
        torch_total_12 = torch.dot(random_momenta_1.view(-1), torch_convolve_gradient_12.contiguous().view(-1))

        [keops_dcp_1, keops_dcp_2, keops_dmom_1, keops_dmom_2] = torch.autograd.grad(
            keops_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])
        [torch_dcp_1, torch_dcp_2, torch_dmom_1, torch_dmom_2] = torch.autograd.grad(
            torch_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])

        # Convert back to numpy.
        keops_convolve_gradient_11 = keops_convolve_gradient_11.detach().cpu().numpy()
        torch_convolve_gradient_11 = torch_convolve_gradient_11.detach().cpu().numpy()
        keops_convolve_gradient_11_bis = keops_convolve_gradient_11_bis.detach().cpu().numpy()
        torch_convolve_gradient_11_bis = torch_convolve_gradient_11_bis.detach().cpu().numpy()
        keops_convolve_gradient_12 = keops_convolve_gradient_12.detach().cpu().numpy()
        torch_convolve_gradient_12 = torch_convolve_gradient_12.detach().cpu().numpy()
        keops_dcp_1 = keops_dcp_1.detach().cpu().numpy()
        keops_dcp_2 = keops_dcp_2.detach().cpu().numpy()
        keops_dmom_1 = keops_dmom_1.detach().cpu().numpy()
        keops_dmom_2 = keops_dmom_2.detach().cpu().numpy()
        torch_dcp_1 = torch_dcp_1.detach().cpu().numpy()
        torch_dcp_2 = torch_dcp_2.detach().cpu().numpy()
        torch_dmom_1 = torch_dmom_1.detach().cpu().numpy()
        torch_dmom_2 = torch_dmom_2.detach().cpu().numpy()

        # Check for equality.
        self.assertTrue(np.allclose(keops_convolve_gradient_11_bis, keops_convolve_gradient_11_bis, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(torch_convolve_gradient_11_bis, torch_convolve_gradient_11_bis, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_convolve_gradient_11, torch_convolve_gradient_11, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_convolve_gradient_12, torch_convolve_gradient_12, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dcp_1, torch_dcp_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dcp_2, torch_dcp_2, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dmom_1, torch_dmom_1, rtol=self.precision, atol=self.precision))
        self.assertTrue(np.allclose(keops_dmom_2, torch_dmom_2, rtol=self.precision, atol=self.precision))
