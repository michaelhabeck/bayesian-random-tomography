"""
Module implementing Hamiltonian Monte Carlo.
"""
import numpy as np

from .probability import Probability
from .mcmc import AdaptiveWalk


class Hamiltonian(object):
    """Hamiltonian
    
    Implements kinetic and potential energies and their gradients with respect
    to positions and momenta. 
    """
    def __init__(self, model):
        """Initialize with probabilistic model.
        
        Parameters
        ----------
        model : instance of Probability
            Probabilistic model that will be sampled with Hamiltonian Monte
            Carlo. 
        """
        if not isinstance(model, Probability):
            raise TypeError('Expected Probability')
        self.model = model

    def set_coords(self, q):
        self.model.params['coordinates'].set(q)

    def potential_energy(self, q):
        """Minus log probability. """
        self.set_coords(q)
        return -self.model.log_prob()

    def kinetic_energy(self, p):
        """Standard kinetic energy assuming unit masses. """
        return 0.5 * np.dot(p, p)

    def gradient_momenta(self, p):
        """Gradient of kinetic energy with respect to momenta. """
        return p

    def gradient_positions(self, q):
        """Gradient of potential energy with respect to positions. """
        self.model.params['forces'].set(0.)        
        self.set_coords(q)
        self.model.update_forces()        
        return -self.model.params['forces'].get().copy()

    def sample_momenta(self, q):
        """Resampling momenta from a standard normal distribution. """
        return np.random.standard_normal(np.shape(q))

    
class State(object):
    """State

    A state in phase space.
    """
    def __init__(self, state, energies):

        positions, momenta = state
        potential_energy, kinetic_energy = energies

        self.positions = positions
        self.momenta   = momenta

        self.potential_energy = potential_energy
        self.kinetic_energy   = kinetic_energy

    @property
    def value(self):
        return self.positions #, self.momenta

    @property
    def log_prob(self):
        return - (self.potential_energy + self.kinetic_energy)

    
class Leapfrog(object):
    """Leapfrog

    Velocity Verlet algorithm.
    """
    def __init__(self, hamiltonian, stepsize=1e-3, n_steps=100):
        """Initialization of Leapfrog algorithm. 
        
        Parameters
        ----------
        hamiltonian : instance of Hamiltonian
            Hamiltonian defining the dynamics. 
        stepsize : positive float
            Step-size parameter.
        n_steps : positive int
            Number of leapfrog steps. 
        """
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError('Expected instance of Hamiltonian')
        if not isinstance(stepsize, float) or stepsize <= 0.:
            raise TypeError('Step-size should be positive float.')
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise TypeError('Number of leapfrog steps should be positive int.')
        self.hamiltonian = hamiltonian        
        self.stepsize = stepsize
        self.n_steps = n_steps
        
    def run(self, positions, momenta):
        """Leapfrog integration. """
        q, p = positions, momenta
        grad_q = self.hamiltonian.gradient_positions
        grad_p = self.hamiltonian.gradient_momenta
        eps = self.stepsize
        
        # first half step for the momenta
        p -= 0.5 * eps * grad_q(q)

        for _ in range(self.n_steps-1):
            q += eps * grad_p(p)
            p -= eps * grad_q(q)

        # last full/half step
        q += eps * grad_p(p)
        p -= 0.5 * eps * grad_q(q)
        
        return q, p

    
class HamiltonianMonteCarlo(AdaptiveWalk):
    """HamiltonianMonteCarlo
    
    Implementation of Hamiltonian Monte Carlo using an adaptive step-size. 
    """
    def __init__(self, model, stepsize=1e-3):
        """Initialize Hamiltonian Monte Carlo algorithm. 
        """
        # need to do this first, because required by create_state
        self.leapfrog = Leapfrog(Hamiltonian(model))

        super(HamiltonianMonteCarlo, self).__init__(
            model, model.params['coordinates'], stepsize=stepsize
        )
        self.uprate = 1.05
        self.downrate = 0.96
        
    def create_state(self):
        """Creates a state from current configuration and generates random
        momenta.
        """
        hamiltonian = self.leapfrog.hamiltonian        
        positions = self.parameter.get().copy()
        momenta = hamiltonian.sample_momenta(positions)        

        # TODO: there is space for speed up by using previously calculated log
        # prob value
        potential_energy = hamiltonian.potential_energy(positions)
        kinetic_energy = hamiltonian.kinetic_energy(momenta)

        return State((positions, momenta), (potential_energy, kinetic_energy))

    def propose(self, state):
        """Propose new state (in phase space) by sampling new momenta and 
        running the leapfrog algorithm.
        """
        hamiltonian = self.leapfrog.hamiltonian

        # resample momenta        
        q = state.positions
        p = hamiltonian.sample_momenta(q)

        state.kinetic_energy = hamiltonian.kinetic_energy(p)
        state.momenta = p

        # run leapfrog
        Q, P = self.leapfrog.run(q.copy(), p.copy())

        V = hamiltonian.potential_energy(Q)
        K = hamiltonian.kinetic_energy(P)

        return State((Q, P), (V, K))
        
    @property
    def stepsize(self):
        """Step-size of the leapfrog integrator. """
        return self.leapfrog.stepsize

    @stepsize.setter
    def stepsize(self, value):
        """Step-size of the leapfrog integrator. """
        self.leapfrog.stepsize = float(value)

