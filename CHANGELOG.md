# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

## [4.3.0] - 2020-08-20
- Updates:
    - pytorch to 1.6
    - pykeops to 1.4.1



## [4.2.0] - 2019-04-15
=======
- Improved deformetrica package importing. Example: `import deformetrica as dfca`
- Bugfix: GUI toggle button values
- Updates:
    - pytorch to 1.1.0
    - pykeops to 1.1.1

## [4.2.0] - 2019-04-18
>>>>>>> dev
- Use a more efficient `_squared_distances` method in `AbstractKernel`. This highly increases performance. contributes to #39
- Improve TorchKernel
- Upgrades: Closes #45
    - pytorch to 1.0.1
    - pykeops to 1.0.1
    - numpy to 1.16
    - nibabel to 2.3.3
    - matplotlib to 3.0
    - pillow to 5.4
    - psutil to 5.4
- Better multiprocessing. high increases in performance. contributes to #39
- Multi-GPU support. contributes to #39
- Better OMP_NUM_THREADS management
- Fix image intensity normalization
- Add a check when working with meshes, to detect null area. fixes #41
- Add set_seed() method to the API
- Bugfix: Multiprocessing pool cleanup
- Python 3.7 compliant. Closes #40
- add `dtype` option to `model_options` in api and xml configuration file
- re-activate keops kernel on macos
- Bugfix: problem with image deformation
- Smarter kernel factory. Closes #37
- Better logger with console and file output. Closes #54
- Functional tests now pass with 1e-10 precision
- Added the longitudinal atlas model, along with functional tests. Closes #51.
- Added the MCMC-SAEM estimator, along with its own save-state system, and corresponding functional tests. Closes #19, #32.
- Refactoring of the deformable object class, now more robust.
- New VTK reader, based on the conda-available VTK library (v8). Closes #50.
- Generalized and harmonized use of the logger, along with automatic txt file dump. Closes #54. 
- Renamed `number-of-threads` to `number-of-processes` to better reflect reality
- Replace `use_cuda` boolean with `gpu_mode` enum. Available gpu_modes are: auto, full, none, kernel

## [4.1.0] - 2018-09-14
### Added
- New API: deformetrica can now be instantiated from python
- Automatic dimension detection from input file
- Allow to use different kernel types for the "shoot" and "flow" operations. Exponential class can now specify `shoot_kernel` and/or `flow_kernel`. resolves #10
- Allow device selection when instantiation a kernel.
  'device' kwarg is now available when using the kernel factory and 'device-device' is now available when using the xml configuration file. resolves #13
- New Python (PyQt5) GUI. Allows the configuration and run of a 'deterministic atlas'. This is an alpha-release
- Update Pykeops to version 0.0.14
- Bugfix: Gradients not computed when number_of_processes>1 and tensor_scalar_type is a FloatTensor. resolves #27
- Bugfix: Memory leak when using estimate_longitudinal_atlas due to pytorch's autograd graph. resolves #33

### Changed
- Split CLI commands into 2 groups: estimate and compute. resolves #6


## [4.0.2] - 2018-08-23
- Corrects the sobolev gradient in the multi-object case


## [4.0.1] - 2018-06-27
- Add different polyline VTK format


## [4.0.0] - 2018-06-14
### Added
- Bugfix: version file not found. issue #24
- Easy install with `conda install -c pytorch -c conda-forge -c anaconda -c aramislab deformetrica`, without any manual compilation.
- All existing deformetrica functionalities now work with 2d or 3d gray level images.
- A L-BFGS optimization method can now be used for registration, regression, deterministic and bayesian atlases.
- Gradients are now automagically computed using PyTorch's autograd.
- It is now possible to perform all computations on the gpu through the `use-cuda` option.

### Changed
- C++ is replaced by Python.
- The "exact" kernel is now named "torch"; the "cudaexact" kernel is now named "keops".
- The "deformable-object-type" xml entry is now split in two entries: "deformable-object-type" and "attachment-type". With this renamming, "NonOrientedSurfaceMesh" becomes a "SurfaceMesh" with a "Varifold" attachment (and an "OrientedSurfaceMesh" a "SurfaceMesh" with a "Current" attachment).

### Removed
- The Nesterov scheme for the gradient ascent optimizer (which was named "FastGradientAscent") is not available anymore. L-BFGS is more efficient though!
