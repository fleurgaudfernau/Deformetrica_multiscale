# Multiscale Deformetrica

This code is an extension of the software Deformetrica.
It implements a coarse-to-fine strategy for atlas construction / image registration.

<div align="center"><img src="/digit_ctf_cv_k3_fold_4_colors.gif" width="200"/></div>

## Installation
      
    git clone https://github.com/fleurgaudfernau/Deformetrica_coarse_to_fine

    virtualenv -p python3 deformetrica && source deformetrica/bin/activate

    python3 -m pip install -r Deformetrica_multiscale/requirements.txt

    python3 -m pip install Deformetrica_multiscale/.
    
<hr style="border:1px solid #2b6777"/>

    virtualenv -p python3 deformetrica && source deformetrica/bin/activate
    
    pip install git+https://github.com/fleurgaudfernau/Deformetrica_multiscale.git
## About Deformetrica

Website: [www.deformetrica.org](http://www.deformetrica.org/)

**Deformetrica** is a software for the statistical analysis of 2D and 3D shape data. It essentially computes deformations of the 2D or 3D ambient space, which, in turn, warp any object embedded in this space, whether this object is a curve, a surface, a structured or unstructured set of points, an image, or any combination of them.

_Deformetrica_ comes with three main applications:
- **registration** : estimates the best possible deformation between two sets of objects;
- **atlas construction** : estimates an average object configuration from a collection of object sets, and the deformations from this average to each sample in the collection;
- **geodesic regression** : estimates an object time-series constrained to match as closely as possible a set of observations indexed by time.

_Deformetrica_ has very little requirements about the data it can deal with. In particular, it does __not__ require point correspondence between objects!

## References

Deformetrica relies on a control-points-based instance of the Large Deformation Diffeomorphic Metric Mapping framework, introduced in [\[Durrleman et al. 2014\]](https://linkinghub.elsevier.com/retrieve/pii/S1053811914005205). Are fully described in this article the **shooting**, **registration**, and **deterministic atlas** applications. Equipped with those fundamental building blocks, additional applications have been successively developed:
- the bayesian atlas, described in [\[Gori et al. 2017\]](https://hal.archives-ouvertes.fr/hal-01359423/);
- the geodesic regression, described in [\[Fishbaugh et al. 2017\]](https://www.medicalimageanalysisjournal.com/article/S1361-8415(17)30044-0/fulltext);
- the parallel transport, described in [\[Louis et al. 2018\]](https://www.researchgate.net/publication/319136479_Parallel_transport_in_shape_analysis_a_scalable_numerical_scheme);
- the longitudinal atlas, described in [\[Bône et al. 2018a\]](https://www.researchgate.net/publication/324037371_Learning_distributions_of_shape_trajectories_from_longitudinal_datasets_a_hierarchical_model_on_a_manifold_of_diffeomorphisms) and [\[Bône et al. 2020\]](https://www.researchgate.net/publication/342642363_Learning_the_spatiotemporal_variability_in_longitudinal_shape_data_sets).

[\[Bône et al. 2018b\]](https://www.researchgate.net/publication/327652245_Deformetrica_4_an_open-source_software_for_statistical_shape_analysis) provides a concise reference summarizing those functionalities, with unified notations.

# Archived repositories

- Deformetrica 3: [deformetrica-legacy2](https://gitlab.icm-institute.org/aramislab/deformetrica-legacy2)
- Deformetrica 2.1: [deformetrica-legacy](https://gitlab.icm-institute.org/aramislab/deformetrica-legacy)
