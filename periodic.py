import math
from abc import ABC, abstractmethod

import numpy as np

class PeriodicFunction(ABC):
    """
    Abstract base class for a periodic function.
    
    Attributes:
        period (int): The number of discrete steps the function will have in its defined interval.
    """
    def __init__(self, period: int):
        assert period > 0, "The period must be a positive integer"
        self.period = period
    
    def __call__(self, t: int) -> float:
        """
        Evaluate the periodic function at an integer time t.
        
        This method first maps t to the remainder of the period, then converts
        it to a value in the desired interval using map_time(), and finally evaluates
        the function using evaluate().
        
        Parameters:
            t (int): The time at which to evaluate the function.
        
        Returns:
            float: The function value at time t.
        """
        # Compute the time within the period
        t_mod = t % self.period
        
        # Map the time to the desired interval (e.g., [0, 1])
        x = self._from_timestep(t_mod)
        
        # Evaluate the function at x
        val =  self._evaluate(x)
        assert val >= 0, "The function value must be non-negative"
        return val
    
    def __getitem__(self, t: int) -> float:
        """
        Allow indexing notation as an alias for calling the function.
        
        Parameters:
            t (int): The timestep.
        
        Returns:
            float: The function value at the timestep.
        """
        return self(t)
    
    @abstractmethod
    def _from_timestep(self, t_mod: int) -> float:
        """
        Convert the integer timestep (t_mod) to a float in the desired interval.
        
        For example, if you want the interval to be [0, 1], you might return t_mod / period.
        
        Parameters:
            t_mod (int): The time value within the period.
            
        Returns:
            float: The mapped time in the target interval.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _evaluate(self, x: float) -> float:
        """
        Evaluate the function given the mapped time x.
        
        Parameters:
            x (float): The time mapped to the function's interval.
        
        Returns:
            float: The function value.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_seed(cls, period: int, seed: int):
        """
        Create a new instance of the function with a specific seed.
        
        This method is useful for generating random functions that are reproducible.
        
        Parameters:
            period (int): The period of the function.
            seed (int): The seed for the random number generator.
            
        Returns:
            PeriodicFunction: A new instance of the function.
        """
        raise NotImplementedError("Subclasses must implement this method")

class PeriodicTwoGaussianSum(PeriodicFunction):
    """
    A periodic function composed of two Gaussian peaks.
    
    The function is defined on the interval [0,1] (per period) as:
        f(x) = amplitude1 * exp(-((x - mean1)**2) / (2 * std1**2))
             + amplitude2 * exp(-((x - mean2)**2) / (2 * std2**2))
    
    Attributes:
        period (int): The period (in integer time units) over which the function repeats.
        amplitude1 (float): Amplitude of the first Gaussian.
        mean1 (float): Mean (peak location) of the first Gaussian in [0, 1].
        std1 (float): Standard deviation of the first Gaussian.
        amplitude2 (float): Amplitude of the second Gaussian.
        mean2 (float): Mean (peak location) of the second Gaussian in [0, 1].
        std2 (float): Standard deviation of the second Gaussian.
    """
    def __init__(self, period: int, amplitude1: float, mean1: float, std1: float,
                 amplitude2: float, mean2: float, std2: float):
        super().__init__(period)
        self.amplitude1 = amplitude1
        self.mean1 = mean1
        self.std1 = std1
        self.amplitude2 = amplitude2
        self.mean2 = mean2
        self.std2 = std2
    
    def _from_timestep(self, t_mod: int) -> float:
        """
        Map the time t_mod (an integer in [0, period-1]) to a float in [0, 1].
        
        Parameters:
            t_mod (int): The time within the period.
            
        Returns:
            float: The normalized time in [0, 1].
        """
        return t_mod / self.period
    
    def _evaluate(self, x: float) -> float:
        """
        Evaluate the periodic Gaussian function at the normalized time x.
        
        Parameters:
            x (float): The time in [0, 1].
            
        Returns:
            float: The value of the function.
        """
        gaussian1 = self.amplitude1 * math.exp(-((x - self.mean1)**2) / (2 * self.std1**2))
        gaussian2 = self.amplitude2 * math.exp(-((x - self.mean2)**2) / (2 * self.std2**2))
        return gaussian1 + gaussian2

    @classmethod
    def from_seed(cls, period: int, seed: int):
        """
        Create a new instance of the function with a specific seed.
        
        This method is useful for generating random functions that are reproducible.
        
        Parameters:
            period (int): The period of the function.
            seed (int): The seed for the random number generator.
            
        Returns:
            PeriodicTwoGaussianSum: A new instance of the function.
        """
        # Set the random seed
        np.random.seed(seed)
        
        # Generate random parameters for the Gaussian peaks
        amplitude1 = np.random.uniform(50, 150)
        mean1 = np.random.uniform(0.2, 0.6)
        std1 = np.random.uniform(0.02, 0.08)
        amplitude2 = np.random.uniform(50, 150)
        mean2 = np.random.uniform(0.6, 0.8)
        std2 = np.random.uniform(0.02, 0.08)
        
        # Create a new instance of the function
        return cls(period, amplitude1=amplitude1, mean1=mean1, std1=std1,
                   amplitude2=amplitude2, mean2=mean2, std2=std2)

# Example usage:
if __name__ == "__main__":
    # Create an instance with a period of 10 (integer time steps)
    # The Gaussian peaks are defined in the normalized [0,1] interval.
    f = PeriodicTwoGaussianSum(
        period=10,
        amplitude1=100,
        mean1=0.4,
        std1=0.05,
        amplitude2=120,
        mean2=0.7,
        std2=0.1
    )
    
    # Evaluate the function at several integer time points
    for t in [0, 4, 7, 10, 14]:
        print(f"f({t}) = {f(t)}")
    
    g = PeriodicTwoGaussianSum.from_seed(10, 42)
    for t in [0, 4, 7, 10, 14]:
        print(f"g({t}) = {g(t)}")
