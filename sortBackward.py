#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:33:43 2022

@author: mikesha
"""

import jax.numpy as jnp
import jax

@jax.custom_vjp
def customSort(x, **kwargs):
    indices = jnp.argsort(x, **kwargs)
    return x[indices]

@jax.jit
def customSort_fwd(x, **kwargs):
    indices = jnp.argsort(x, **kwargs)
    return (x[indices], kwargs), indices

@jax.jit
def customSort_bwd(val, diffs):
    indices, kwargs = val
    inversePermutation = jnp.argsort(indices, **kwargs)
    return diffs[inversePermutation]

customSort.defvjp(customSort_fwd, customSort_bwd)

from jax import random
key = random.PRNGKey(0)
arr = jax.random.uniform(key, shape=(1000000,))
custom_bwd = jax.vjp(customSort, arr)
normal_bwd = jax.vjp(jnp.sort, arr)

@jax.jit
def normalSort_bwd(arr):
    return jax.vjp(jnp.sort, arr)[1]

@jax.jit
def customSort_bwd(arr):
    return jax.vjp(customSort, arr)[1]

jnp.all((normalSort_bwd(arr)).args[0].args[0][0].flatten() 
        == customSort_bwd(arr).args[0].args[0][0])
