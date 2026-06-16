import jax.numpy as jnp
from typing import Any, NamedTuple, Mapping, Tuple, TypeVar, Union

import numpy as np


class Transition(NamedTuple):
    
    observation: jnp.ndarray
    action : jnp.ndarray
    reward : jnp.ndarray
    discount : jnp.ndarray      # in code it's written as discount 
    next_observation : jnp.ndarray
    extras : Any = () 
    

