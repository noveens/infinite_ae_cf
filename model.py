import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax

def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params['depth'],
        num_classes=hyper_params['num_items']
    )
    # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
    # kernel_fn = nt.batch(kernel_fn, batch_size=128)
    kernel_fn = functools.partial(kernel_fn, get='ntk')

    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        K_train = kernel_fn(X_train, X_train)
        K_predict = kernel_fn(X_predict, X_train)
        K_reg = (K_train + jnp.abs(reg) * jnp.trace(K_train) * jnp.eye(K_train.shape[0]) / K_train.shape[0])     
        return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))

    return kernelized_rr_forward, kernel_fn

def FullyConnectedNetwork( 
    depth,
    W_std = 2 ** 0.5, 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk'
):
    activation_fn = stax.Relu()
    dense = functools.partial(stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth): layers += [dense(1024), activation_fn] 
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization)]

    return stax.serial(*layers)
