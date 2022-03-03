import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import time

# Estimators

from ..core.estimators.mcmc_saem import McmcSaem
from ..core.estimators.scipy_optimize import ScipyOptimize
from ..core.estimators.gradient_ascent import GradientAscent
from ..core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from ..support.utilities.general_settings import Settings
from ..support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from ..core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from ..core.model_tools.manifolds.metric_learning_nets import MnistNet, ScalarNet
from ..core.models.deep_pga import DeepPga
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..core.observations.deformable_objects.image import Image

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Runs the deep pga model on 1000 randomly extracted digits for mnist, for varying latent space dimensions.
# Experimental for now.


def run_model_on_MNIST(latent_space_dimension):
    Settings().dimension = 2
    Settings().output_dir = 'output_' + str(latent_space_dimension)

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True)

    train_dataset = LongitudinalDataset()
    test_dataset = LongitudinalDataset()

    train_labels = []
    test_labels = []

    for elt in trainset:
        image, label = elt
        dmo = DeformableMultiObject()
        image_object = Image()
        image_object.set_intensities(image / np.max(image))
        image_object.set_affine(np.eye(Settings().dimension + 1))
        image_object.intensities_dtype = 'uint8'
        dmo.object_list = [image_object]
        dmo.update()
        if len(train_dataset.deformable_objects) < 200:
            train_dataset.deformable_objects.append([dmo])
            train_dataset.subject_ids.append(len(train_dataset.deformable_objects))
            train_labels.append(label)
        elif len(test_dataset.deformable_objects) < 200:
            test_dataset.deformable_objects.append([dmo])
            test_dataset.subject_ids.append(len(train_dataset.deformable_objects))
            test_labels.append(label)

    np.savetxt(os.path.join(Settings().output_dir, 'labels_train.txt'), np.array(train_labels))
    np.savetxt(os.path.join(Settings().output_dir, 'labels_test.txt'), np.array(test_labels))

    train_dataset.update()
    test_dataset.update()

    train_images_data = np.array([elt[0].object_list[0].get_intensities() for elt in train_dataset.deformable_objects])
    test_images_data = np.array([elt[0].object_list[0].get_intensities() for elt in test_dataset.deformable_objects])

    train_images_data_torch = torch.from_numpy(train_images_data).type(Settings().tensor_scalar_type)
    test_images_data_torch = torch.from_numpy(test_images_data).type(Settings().tensor_scalar_type)

    pca = PCA(n_components=latent_space_dimension)
    latent_space_positions = pca.fit_transform([elt.flatten() for elt in train_images_data])
    reconstructed = pca.inverse_transform(latent_space_positions)

    latent_test = pca.transform([elt.flatten() for elt in test_images_data])
    reconstructed_test = pca.inverse_transform(latent_test)

    reconstruction_error_train = mean_squared_error(reconstructed, [elt.flatten() for elt in train_images_data])
    reconstruction_error_test = mean_squared_error(reconstructed_test, [elt.flatten() for elt in test_images_data])
    logger.info('PCA mse on train:', reconstruction_error_train)
    logger.info('PCA mse on test:', reconstruction_error_train)
    to_write = np.array([reconstruction_error_train, reconstruction_error_test])
    np.savetxt(os.path.join(Settings().output_dir, 'pca_reconstruction_error.txt'), to_write)

    # We now normalize every latent_space_positions
    for i in range(latent_space_dimension):
        latent_space_positions[:, i] = (latent_space_positions[:, i] - np.mean(latent_space_positions[:, i]))/np.std(latent_space_positions[:, i])

    latent_space_positions_torch = torch.from_numpy(latent_space_positions).type(Settings().tensor_scalar_type)

    # We now instantiate the neural network
    net = MnistNet(in_dimension=latent_space_dimension)

    criterion = nn.MSELoss()
    nb_epochs = 50
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    train_dataset_fit = TensorDataset(latent_space_positions_torch, train_images_data_torch)
    train_dataloader = DataLoader(train_dataset_fit, batch_size=10, shuffle=True)

    for epoch in range(nb_epochs):
        train_loss = 0
        nb_train_batches = 0
        for (z, y) in train_dataloader:
            nb_train_batches += 1
            predicted = net(z)
            loss = criterion(predicted, y)
            net.zero_grad()
            loss.backward()
            train_loss += loss.detach().numpy()
            optimizer.step()

        train_loss /= nb_train_batches

        logger.info("Epoch {}/{}".format(epoch, nb_epochs),
              "Train loss:", train_loss)

    noise_variance = train_loss/(28*28)

    model = instantiate_and_estimate_model(train_dataset, latent_space_dimension, latent_space_positions, net, noise_variance)

    register_test_to_model(model, test_dataset, 'test_output_'+str(latent_space_dimension), latent_space_dimension)

def run_on_cylinder():
    Settings().dimension = 3

    Settings().output_dir = 'cylinder_output'

    train_length = 1000
    test_length = 300

    train_coords = np.random.normal(0, 1, size=(train_length, 2))
    test_coords = np.random.normal(0, 1, size=(test_length, 2))

    train_set = np.array([[np.cos(elt[0]), np.sin(elt[0]), elt[1]] for elt in train_coords])
    test_set = np.array([[np.cos(elt[0]), np.sin(elt[0]), elt[1]] for elt in test_coords])

    np.save(os.path.join(Settings().output_dir, 'train_latent_coordinates.npy'), train_coords)
    np.save(os.path.join(Settings().output_dir, 'test_latent_coordinates.npy'), test_coords)

    train_dataset = LongitudinalDataset()
    train_dataset.deformable_objects = [[elt] for elt in train_set]
    train_dataset.subject_ids = [str(elt) for elt in range(train_length)]
    train_dataset.update()

    test_dataset = LongitudinalDataset()
    test_dataset.deformable_objects = [[elt] for elt in test_set]
    test_dataset.subject_ids = [str(elt) for elt in range(test_length)]
    test_dataset.update()

    latent_space_dimension = 2

    # We initialize the lsd positions with pca
    pca = PCA(n_components=latent_space_dimension)
    latent_space_positions = pca.fit_transform(train_set)
    reconstructed_train = pca.inverse_transform(latent_space_positions)

    latent_test = pca.transform([elt.flatten() for elt in test_set])
    reconstructed_test = pca.inverse_transform(latent_test)

    reconstruction_error_train = mean_squared_error(reconstructed_train, train_set)
    reconstruction_error_test = mean_squared_error(reconstructed_test, test_set)
    logger.info('PCA mse on train:', reconstruction_error_train)
    logger.info('PCA mse on test:', reconstruction_error_test)

    # We now initialize the neural network

    # We now normalize every latent_space_positions
    for i in range(latent_space_dimension):
        latent_space_positions[:, i] = (latent_space_positions[:, i] - np.mean(latent_space_positions[:, i])) / np.std(
            latent_space_positions[:, i])

    latent_space_positions_torch = torch.from_numpy(latent_space_positions).type(Settings().tensor_scalar_type)
    train_set_torch = torch.from_numpy(train_set).type(Settings().tensor_scalar_type)

    # We now instantiate the neural network
    net = ScalarNet(in_dimension=latent_space_dimension, out_dimension=3)

    criterion = nn.MSELoss()
    nb_epochs = 1
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    train_dataset_fit = TensorDataset(latent_space_positions_torch, train_set_torch)
    train_dataloader = DataLoader(train_dataset_fit, batch_size=10, shuffle=True)

    for epoch in range(nb_epochs):
        train_loss = 0
        nb_train_batches = 0
        for (z, y) in train_dataloader:
            nb_train_batches += 1
            predicted = net(z)
            loss = criterion(predicted, y)
            net.zero_grad()
            loss.backward()
            train_loss += loss.detach().numpy()
            optimizer.step()

        train_loss /= nb_train_batches

        logger.info("Epoch {}/{}".format(epoch, nb_epochs),
              "Train loss:", train_loss)

    noise_variance = train_loss / (28 * 28)

    model = instantiate_and_estimate_model(train_dataset, latent_space_dimension, latent_space_positions, net,
                                           noise_variance)

    return


def register_test_to_model(model, test_dataset, output_dir, latent_space_dimension):
    Settings().output_dir = output_dir
    if not os.path.isdir(Settings().output_dir):
        os.mkdir(Settings().output_dir)

    # We now need to estimate the residual on the test set... we create a new estimator.
    model.is_frozen['metric_parameters'] = True
    model.is_frozen['noise_variance'] = True
    estimator = ScipyOptimize()
    estimator.memory_length = 10

    estimator.convergence_tolerance = 1e-3
    estimator.max_line_search_iterations = 10

    estimator.max_iterations = 300

    estimator.print_every_n_iters = 1
    estimator.save_every_n_iters = 10

    estimator.dataset = test_dataset
    estimator.statistical_model = model

    individual_RER = {}
    individual_RER['latent_position'] = np.zeros((200, latent_space_dimension))
    estimator.individual_RER = individual_RER

    estimator.update()
    estimator.write()

def instantiate_and_estimate_model(dataset, latent_space_dimension, initial_lds_positions, initial_net, initial_noise_variance):
    """
    isntantiate the deep pga model and runs it on the data with the initialized arguments
    """
    model = DeepPga()

    model.template = dataset.deformable_objects[0][0]

    metric_parameters = initial_net.get_parameters()
    model.net = initial_net
    model.set_metric_parameters(metric_parameters)

    latent_space_dimension = latent_space_dimension

    model.latent_space_dimension = latent_space_dimension

    model.set_noise_variance(initial_noise_variance)
    model.update()

    model.individual_random_effects['latent_position'].mean = np.zeros((latent_space_dimension,))
    model.individual_random_effects['latent_position'].set_variance(1.0)

    individual_RER = {}
    individual_RER['latent_position'] = initial_lds_positions

    sampler = SrwMhwgSampler()
    estimator = McmcSaem()
    estimator.sampler = sampler

    # latent positions proposal:
    latent_position_proposal_distribution = MultiScalarNormalDistribution()
    latent_position_proposal_distribution.set_variance_sqrt(0.1)
    sampler.individual_proposal_distributions['latent_position'] = latent_position_proposal_distribution

    estimator.sample_every_n_mcmc_iters = 15

    estimator.max_iterations = 200
    estimator.number_of_burn_in_iterations = 200
    estimator.max_line_search_iterations = 10
    estimator.convergence_tolerance = 1e-3

    estimator.print_every_n_iters = 1
    estimator.save_every_n_iters = 1

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'DeepPgaModel'
    logger.info('')
    logger.info('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    logger.info('>> Estimation took: ' + str(time.strftime("%d days, %H hours, %M minutes and %S seconds.",
                                                         time.gmtime(end_time - start_time))))

    return model

if __name__ == '__main__':
    #run_model_on_MNIST(2)

    run_on_cylinder()






