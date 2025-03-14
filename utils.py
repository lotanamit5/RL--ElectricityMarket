from torch.distributions import Normal


class NormalNoiseWrapper():
    def __init__(self, func: callable,
                 loc=0, scale=1):
        assert callable(func)
        self.func = func
        self.noise = Normal(loc=loc, scale=scale)
    
    def __call__(self, *args, **kwds):
        return self.func(*args) + self.noise.sample().item()
