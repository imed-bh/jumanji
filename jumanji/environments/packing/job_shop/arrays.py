from typing import Any

import jax.numpy as jnp
import chex


def decrement(array: chex.Array, stop_value: Any = 0):
    return jnp.where(array > stop_value, array - 1, stop_value)