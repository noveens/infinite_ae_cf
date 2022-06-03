import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import copy
import random
import numpy as np

from utils import log_end_epoch, get_item_propensity, get_common_path

def train(hyper_params, data):
    from model import make_kernelized_rr_forward
    from eval import evaluate

    # This just instantiates the function
    kernelized_rr_forward, kernel_fn = make_kernelized_rr_forward(hyper_params)
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    '''
    NOTE: No training required! We will compute dual-variables \alpha on the fly in `kernelized_rr_forward`
          However, if we needed to perform evaluation multiple times, we could pre-compute \alpha like so:
    
    import jax, jax.numpy as jnp, jax.scipy as sp
    @jax.jit
    def precompute_alpha(X, lamda=0.1):
        K = kernel_fn(X, X)
        K_reg = (K + jnp.abs(lamda) * jnp.trace(K) * jnp.eye(K.shape[0]) / K.shape[0])
        return sp.linalg.solve(K_reg, X, sym_pos=True)
    alpha = precompute_alpha(sampled_matrix, lamda=0.1) # Change for the desired value of lamda
    '''

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)
    
    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@100"
    best_metric, best_lamda = None, None

    # Validate on the validation-set
    for lamda in [ 0.0, 1.0, 5.0, 20.0, 50.0, 100.0 ] if hyper_params['grid_search_lamda'] else [ hyper_params['lamda'] ]:
        hyper_params['lamda'] = lamda
        val_metrics = evaluate(hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix)
        if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric): best_metric, best_lamda = val_metrics[VAL_METRIC], lamda

    # Return metrics with the best lamda on the test-set
    hyper_params['lamda'] = best_lamda
    test_metrics = evaluate(hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix, test_set_eval = True)
    
    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics

def main(hyper_params, gpu_id = None):
    if gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax.config import config
    if 'float64' in hyper_params and hyper_params['float64'] == True: config.update('jax_enable_x64', True)

    from data import Dataset

    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    os.makedirs("./results/logs/", exist_ok=True)
    hyper_params['log_file'] = "./results/logs/" + get_common_path(hyper_params) + ".txt"
    
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params) # Updated w/ data-stats

    return train(hyper_params, data)

if __name__ == "__main__":
    from hyper_params import hyper_params
    main(hyper_params)
