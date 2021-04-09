import numpy as np

from scipy.special import logsumexp

from .mcmc import (
    AdaptiveWalk as _AdaptiveWalk,
)

from .hmc import (
    HamiltonianMonteCarlo as _HamiltonianMonteCarlo
)


class AdaptiveWalk(_AdaptiveWalk):
    """Adaptive Walk

    keeping track of the stepsizes...
    """
    stepsizes = []

    @property
    def stepsize(self):
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        self._stepsize = float(value)
        self.stepsizes.append(self.stepsize)


class RotationSampler(AdaptiveWalk):

    quaternions = np.load('../data/quaternions.npz')
    quaternions = [quaternions['level{}'.format(level)]
                   for level in range(3)]

    
    def __init__(self, image, stepsize=1e-2, level=-1):
        rotation = image.mock._rotation
        super(RotationSampler, self).__init__(
            image, rotation, stepsize, adapt_until=int(1e2)
        )
        self.activate()
        self.level = int(level)
        assert self.level in [-1, 0, 1]

        
    def sample_initial(self):
        """Systematic scan. """
        if self.level < 0: return

        # evaluate log probability for all quaternions in the 600-cell
        log_prob = []
        quaternions = RotationSampler.quaternions[self.level]
        quaternions = np.vstack([self.parameter.get().copy().reshape(1, -1),
                                 quaternions])
        for q in quaternions:
            self.parameter.set(q)
            self.model.update()
            log_prob.append(self.model.log_prob())
        log_prob = np.array(log_prob) - logsumexp(log_prob)
        prob = np.exp(log_prob)

        i = np.random.choice(np.arange(len(prob)), p=prob)
        q = quaternions[i]

        self.parameter.set(q)
        self.state = self.create_state()
        
    
    def sample(self, n_steps=10, resample_frequency=0.):
        """Metropolis-Hastings with an occasional systematic scan. """
        self.level = np.random.choice(
            [0, -1],
            p=[resample_frequency, 1-resample_frequency]
        )
        self.sample_initial()
        samples = []
        while len(samples) < n_steps:
            samples.append(next(self))
        return samples
    

class HamiltonianMonteCarlo(_HamiltonianMonteCarlo):

    def next(self):
        result = super(HamiltonianMonteCarlo, self).next()
        # print some info
        if len(self.history) and not len(self.history) % 20:
            print('{0}, stepsize = {1:.3e}, -log_prob = {2:.3e}'.format(
                self.history, self.stepsize, self.state.potential_energy)
            )
        return result

    
