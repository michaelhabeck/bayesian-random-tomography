"""
Collection of priors.
"""
import numpy as np

from .probability import Probability
from .params import Scale, Location

class BoltzmannEnsemble(Probability):
    """BoltzmannEnsemble
    
    TODO
    """
    def __init__(self, name, forcefield, params):
        """
        Parameters
        ----------
        name : str
            Name of Boltzmann ensemble

        forcefield : 
        """
        super(BoltzmannEnsemble, self).__init__(name, params)

        self.forcefield = forcefield

        # inverse temperature
        self._beta = Scale(self.name + '.beta')
        self.params.add(self._beta)

        # local copy of Cartesian gradient
        self._forces = np.zeros_like(self.params['coordinates'].get())

        
    def log_prob(self):
        if np.isclose(self.beta, 0):
            return 0.
        else:
            coords = self.params['coordinates'].get()
            return - self.beta * self.forcefield.energy(coords)

        
    # TODO: ?
    # for consistency with likelihoods
    def update(self):
        pass

    
    def update_forces(self):

        if np.isclose(self.beta, 0):
            return
        
        coords = self.params['coordinates'].get()

        self._forces[...] = 0.

        self.forcefield.update_list(coords)        
        self.forcefield.update_gradient(coords, self._forces)

        self._forces *= -self.beta
        
        self.params['forces']._value += self._forces

    @property
    def beta(self):
        """Inverse temperature. """
        return self._beta.get()

    @beta.setter
    def beta(self, value):
        self._beta.set(value)


        
class TsallisEnsemble(BoltzmannEnsemble):

    def __init__(self, name, forcefield, params):

        super(TsallisEnsemble, self).__init__(name, forcefield, params)

        self._q = Scale(self.name + '.q')
        self.params.add(self._q)

        self._E_min = Location(self.name + '.E_min')
        self.params.add(self._E_min)

    def log_prob(self):
        if self.q < 1+1e-10:
            return super(TsallisEnsemble, self).log_prob()
        else:
            coords = self.params['coordinates'].get()            
            E = self.beta * self.forcefield.energy(coords)
            q = self.q
            E_min = self.beta * self.E_min            
            return - q / (q-1) * np.log(1 + (q-1) * (E-E_min)) - E_min

    def update_forces(self):

        if self.q < 1+1e-10:
            super(TsallisEnsemble, self).update_forces()
        else:
            coords = self.params['coordinates'].get()
            self._forces[...] = 0.

            self.forcefield.update_list(coords)        
            E = self.beta * self.forcefield.update_gradient(
                coords, self._forces
            )
            q  = self.q
            E_min = self.beta * self.E_min
            f = - self.beta * q / (1 + (q-1) * (E-E_min))
        
            self._forces *= f

            self.params['forces']._value += self._forces

        
    @property
    def q(self):
        """Tsallis parameter. """
        return self._q.get()

    @q.setter
    def q(self, value):
        value = float(value)
        if value < 1.:
            raise ValueError('Tsallis q must be >= 1')
        self._q.set(value)

    @property
    def E_min(self):
        """Minimum energy. """
        return self._E_min.get()

    @E_min.setter
    def E_min(self, value):
        self._E_min.set(value)

