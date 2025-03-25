
import torch
from torch.distributions import Normal
from contextlib import contextmanager

class NormalNoiseWrapper():
    def __init__(self, func: callable,
                 loc=0, scale=1):
        assert callable(func)
        self.func = func
        self.noise = Normal(loc=loc, scale=scale)
        self.random_state = torch.randint(0, 10000, (1,)).item()
    
    def __call__(self, t):
        with torch_temporary_seed(self.random_state + t):
            noise_val = self.noise.sample().item()
        return max(self.func(t) + noise_val, self.func(t) * 0.1)


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