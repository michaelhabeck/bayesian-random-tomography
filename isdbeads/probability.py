from .core import Nominable
from .params import Parameters


# TODO: make abstract class
class Probability(Nominable):
    """Probability

    Generic class to be subclassed by all probabilistic models.
    """
    def __init__(self, name, params=None):
        self.name = name
        self.params = params if params is not None else Parameters()
        
    def log_prob(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

        
