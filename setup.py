#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from glob import glob
from os.path import splitext, basename

from setuptools import setup, find_packages


def read(*parts):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, *parts)) as fp:
        return fp.read().strip()


def find_version(*file_paths):
    import re
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("deformetrica", "__init__.py")

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


def str_to_bool(s):
    if s is None:
        return False

    assert isinstance(s, str), 'given argument must be a string'
    if s.lower() in ['true', '1', 'yes']:
        return True
    elif s.lower() in ['false', '0', 'no']:
        return False
    else:
        raise LookupError


# build gui by default
build_gui = str_to_bool(os.environ['BUILD_GUI']) if 'BUILD_GUI' in os.environ else True

print('Building Deformetrica version ' + version + ', BUILD_GUI=' + str(build_gui))


def build_deformetrica():
    print('build_deformetrica()')
    setup(
        name='deformetrica',
        version=version,
        url='http://www.deformetrica.org',
        description='Software for the statistical analysis of 2D and 3D shape data.',
        long_description=open('README.md', encoding='utf-8').read(),
        author='Fleur Gaudfernau',
        maintainer='Deformetrica developers',
        maintainer_email='fleur.gaudfernau@etu.u-paris.fr',
        license='INRIA license',
        package_dir={'deformetrica': './deformetrica'},
        packages=find_packages(exclude=['build*', 'docs*', 'examples*', 'output*', 'sandbox*', 'utilities*', 'tests*', '.*']),
        py_modules=[splitext(basename(path))[0] for path in glob('deformetrica/*.py')],     # needed to include base __init__.py and __main__.py
        package_data={'': ['*.json', '*.png']},
        include_package_data=True,
        data_files=[('deformetrica', ['LICENSE.txt'])],
        zip_safe=False,
        entry_points={
            'console_scripts': ['deformetrica=deformetrica.__main__:main'],  # CLI
        },
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: X11 Applications :: Qt',
            'Operating System :: OS Independent',
            'Intended Audience :: End Users/Desktop',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Software Development :: Libraries'
        ],
        install_requires=[
            'cachetools>=4.2.4',
            'numpy>=1.16.2',
            'scikit-learn>=0.20.3',
            'matplotlib>=2.2.2',
            'nibabel>=2.3.3',
            'pillow>=5.4.1',
            'torch==1.6',
            'torchvision==0.7',
            'psutil>=5.4.8',
            'vtk>=8.2.0',
            'pykeops==1.0b1',
            'PyQt5'
        ],
        extra_link_args=['-Wl,-headerpad_max_install_names']
    )


def build_deformetrica_nox():
    print('build_deformetrica_nox()')
    setup(
        name='deformetrica-nox',
        version=version,
        url='http://www.deformetrica.org',
        description='Software for the statistical analysis of 2D and 3D shape data.',
        long_description=open('README.md', encoding='utf-8').read(),
        author='Fleur Gaudfernau',
        maintainer='Deformetrica developers',
        maintainer_email='fleur.gaudfernau@etu.u-paris.fr',
        license='INRIA license',
        package_dir={'deformetrica': './deformetrica'},
        packages=find_packages(exclude=['gui*', 'build*', 'docs*', 'examples*', 'output*', 'sandbox*', 'utilities*', 'tests*', '.*']),  # exclude gui
        py_modules=[splitext(basename(path))[0] for path in glob('deformetrica/*.py')],     # needed to include base __init__.py and __main__.py
        package_data={'': ['*.json', '*.png']},
        include_package_data=True,
        data_files=[('deformetrica', ['LICENSE.txt'])],
        zip_safe=False,
        entry_points={
            'console_scripts': ['deformetrica=deformetrica.__main__:main'],  # CLI
        },
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: End Users/Desktop',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Software Development :: Libraries'
        ],
        install_requires=[
            'cachetools>=4.2.4',
            'numpy>=1.16.2',
            'scikit-learn>=0.20.3',
            'matplotlib>=2.2.2',
            'nibabel>=2.3.3',
            'pillow>=5.4.1',
            'torch==1.4',
            'torchvision==0.5',
            'psutil>=5.4.8',
            'vtk>=8.2.0',
            'pykeops==1.0b1',
            'PyQt5'
        ],
        extra_link_args=['-Wl,-headerpad_max_install_names']
    )


if build_gui:
    build_deformetrica()
else:
    build_deformetrica_nox()
