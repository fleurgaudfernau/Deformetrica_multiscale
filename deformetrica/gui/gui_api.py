import threading

import deformetrica as dfca

import logging
global logger
logger = logging.getLogger()


def call(name, params, estimator_callback=None, callback=lambda *args: None):
    # global running
    logger.info(params)

    def func():
        # global running
        try:
            deformetrica_command_to_run = globals()[name]
            ret = deformetrica_command_to_run(**dict(params), estimator_callback=estimator_callback)
            callback(ret)
            return ret
        except Exception as e:
            logger.info("An error has occurred : " + str(e))
            callback(None)
        return None

    t = threading.Thread(target=func)
    t.daemon = True
    t.start()


corresp = {
    "kernel": {
        "torch": dfca.kernels.Type.TORCH,
        "keops": dfca.kernels.Type.KEOPS
    },

    "optimize": {
        "GradientAscent": dfca.estimators.GradientAscent,
        "ScipyLBFGS": dfca.estimators.ScipyOptimize
    }
}


def estimate_deterministic_atlas(deformation_parameters, template, optimization_parameters, estimator_callback, **kwargs):
    """
    Format parameters from GUI into a Deformetrica comprehensive format
    """

    with dfca.Deformetrica(output_dir=deformation_parameters["output_dir"]
                           if "output_dir" in deformation_parameters
                           else './output', verbosity="INFO") as deformetrica:

        template_specifications = {}
        dataset_specifications = {}

        # template_specifications
        i = 0
        subject_count = len(template[0]["filenames"])
        for temp in template:
            rtemp = dict(temp)
            subject_count = subject_count if len(temp["filenames"]) > subject_count else len(temp["filenames"])
            template_specifications[str(i)] = rtemp
            i += 1

        # dataset_specifications
        visit_ages = []
        subject_ids = []
        dataset_file_names = []

        for subjID in range(subject_count):
            visit_ages.append([])
            subject_ids.append(str(subjID))
            obj = {}
            for j in range(i):
                obj[str(j)] = template_specifications[str(j)]["filenames"][subjID]
            dataset_file_names.append([obj])

        dataset_specifications['dataset_filenames'] = dataset_file_names
        dataset_specifications['subject_ids'] = subject_ids

        # run actual deformetrica function
        deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={**optimization_parameters, 'callback': estimator_callback},
            model_options={**deformation_parameters})

