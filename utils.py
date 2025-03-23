
import torch
from torch.distributions import Normal
from contextlib import contextmanager


@contextmanager
def torch_temporary_seed(seed: int):
    """
    A context manager which temporarily sets torch's random seed, then sets the random
    number generator state back to its previous state.
    :param seed: The temporary seed to set.
    """
    if seed is None:
        yield
    else:
        state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(state)

class NormalNoiseWrapper():
    def __init__(self, func: callable,
                 loc=0, scale=1):
        assert callable(func)
        self.func = func
        self.noise = Normal(loc=loc, scale=scale)
    
    def __call__(self, *args, **kwds):
        val =  self.func(*args) + self.noise.sample().item()
        if val < 0:
            val = 1e-3
        return val

