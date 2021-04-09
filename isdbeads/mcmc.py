"""
Markov chain Monte Carlo sampling
"""
import numpy as np

from copy import deepcopy
from collections import OrderedDict

from .probability import Probability
from .params import Parameter, Parameters


class History(object):
    """History

    Keeping track of accepted / rejected Monte Carlo trials.
    """
    def __init__(self):
        self.clear()

    def __len__(self):
        return len(self._history)

    def __getitem__(self, index):
        return self._history[index]

    def update(self, accept):
        self._history.append(int(accept))

    def clear(self):
        self._history = []

    def acceptance_rate(self, burnin=0):
        return np.mean(self._history[burnin:])

    def __str__(self):
        return 'n_steps = {0}, acceptance rate = {1:.1%}'.format(
            len(self), self.acceptance_rate()
        )

    
class State(object):
    """State

    Sample generated with MCMC.
    """
    def __init__(self, value, log_prob):
        self.value = value if not np.iterable(value) else value.copy()
        self.log_prob = log_prob

        
class MetropolisHastings(object):
    """MetropolisHastings

    Generic Metropolis-Hastings algorithm implemented as a generator. 
    Subclasses need to implement the `propose` method. 
    """
    def __init__(self, model, parameter):
        """Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        model : instance of Probability
            Conditional posterior distribution of the parameter.
        parameter : instance of Parameter
            Parameter that will be sampled by the MH algorithm. 
        """
        if not isinstance(model, Probability):
            raise TypeError('Expected Probability')
        if not isinstance(parameter, Parameter):
            raise TypeError('Expected Parameter')
        self.model = model
        self.parameter = parameter
        self.history = History()
        self.history.clear()
        self.state = self.create_state()
        
    def create_state(self):
        """Create a state from current parameter settings. """
        self.model.update()
        return State(self.parameter.get(), self.model.log_prob())

    def propose(self, state):
        """Generates a new state from a given input state. """
        raise NotImplementedError

    def next(self):
        """Proposes a new value for the parameter which is accepted or rejected
        according to the Metropolis criterion. 
        """
        current = self.state
        candidate = self.propose(current)

        diff = candidate.log_prob - current.log_prob
        accept = np.log(np.random.random()) < diff

        self.state = candidate if accept else current 

        self.history.update(accept)

        return self.state

    def __iter__(self):
        return self

    
class RandomWalk(MetropolisHastings):
    """RandomWalk
    
    Random-walk Metropolis-Hastings algorithm using uniform proposals scaled
    with a step-size parameter. 
    """
    def __init__(self, model, parameter, stepsize=1e-1):
        """Random-walk Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        model : instance of Probability
            Conditional posterior distribution of the parameter.
        parameter : instance of Parameter
            Parameter that will be sampled by the MH algorithm. 
        stepsize : positive float
            Step-size parameter. 
        """
        if not isinstance(stepsize, float) or stepsize <= 0.:
            raise TypeError('Step-size should be positive float.')
        super(RandomWalk, self).__init__(model, parameter)
        self.stepsize = float(stepsize)

    def propose(self, state):
        """Uniform proposal distribution. """
        x = state.value
        y = np.random.uniform(x-self.stepsize, x+self.stepsize)
        self.parameter.set(y)
        return self.create_state()

    
class AdaptiveWalk(RandomWalk):
    """AdaptiveWalk
    
    Random-walk Metropolis-Hastings algorithm with an adaptive step-size
    parameter. 
    """    
    def __init__(self, model, parameter, stepsize=1e-1,
                 uprate=1.02, downrate=0.98, adapt_until=0):

        super(AdaptiveWalk, self).__init__(model, parameter, stepsize)

        self.uprate      = float(uprate)
        self.downrate    = float(downrate)
        self.adapt_until = int(adapt_until)

        self.deactivate()

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    @property
    def is_active(self):
        return self._active

    def next(self):

        state = super(AdaptiveWalk, self).next()

        if len(self.history) >= self.adapt_until:
            self.deactivate()

        if self.is_active:
            self.stepsize *= self.uprate if self.history[-1] else self.downrate

        return state


class Ensemble(object):
    """Ensemble

    Storing samples generated with MCMC. 
    """
    def __init__(self, params):
        self._values = OrderedDict()
        if isinstance(params, Parameters):
            for param in params:
                self._values[param.name] = []
        elif isinstance(params, list) or isinstance(params, tuple):
            for param in params:
                self._values[param] = []
        self._size = 0
        
    def update(self, params):
        if not isinstance(params, Parameters):
            raise TypeError('Parameters expected')
        for param in params:
            if param.name in self._values:
                self._values[param.name].append(np.copy(param.get()))
        self._size += 1
        
    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def __getitem__(self, param_name):
        return np.array(self._values[param_name])
        
    def __len__(self):
        return self._size
