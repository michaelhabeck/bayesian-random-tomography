"""
Non-bonded force fields.

PROLSQ uses a quartic repulsion term to penalize particle-particle
clashses.

ROSETTA is a linearly ramped Lennard-Jones potential.
"""
import numpy as np

from ._isd import prolsq, rosetta
from .core import ctypeproperty, CWrapper, Nominable
from .nblist import NBList
from collections import namedtuple


class Forcefield(Nominable, CWrapper):
    """Forcefield

    Non-bonded force field enforcing volume exclusion. 
    """
    def __init__(self, name):
        self.init_ctype()
        self.set_default_values()
        self.name = name

    def __getstate__(self):
        state = super(Forcefield, self).__getstate__()
        state['nblist'] = state.pop('_nblist')
        state['n_types'] = self.n_types
        state['d'] = self.d
        state['k'] = self.k        
        return state
    
    def set_default_values(self):
        self.nblist = None        
        self.enable()

    def is_enabled(self):
        return self.ctype.enabled == 1

    def enable(self, enabled=True):
        self.ctype.enabled = int(enabled)

    def disable(self):
        self.enable(0)

    def update_list(self, coords):
        """Update neighbor list. """
        self.ctype.nblist.update(coords.reshape(-1, 3), 1)
        
    def energy(self, coords, update=True):
        if update:
            self.update_list(coords)
        return self.ctype.energy(coords.reshape(-1, 3), self.types)

    def update_gradient(self, coords, forces):
        return self.ctype.update_gradient(coords, forces, self.types, 1)

    def __str__(self):
        s = '{0}(n_types={1:.2f})'        
        return s.format(self.__class__.__name__, self.n_types)

    __repr__ = __str__

    @ctypeproperty(np.array)
    def k(): pass

    @ctypeproperty(np.array)
    def d(): pass

    @ctypeproperty(int)
    def n_types(): pass

    @property
    def nblist(self):
        return self._nblist

    @nblist.setter
    def nblist(self, value):        
        self._nblist = value
        if value is not None:
            self.ctype.nblist = value.ctype

    
class PROLSQ(Forcefield):

    def __init__(self, name='PROLSQ'):
        super(PROLSQ, self).__init__(name)

    def init_ctype(self):
        self.ctype = prolsq()

        
class ROSETTA(Forcefield):

    def __init__(self, name='ROSETTA'):
        super(ROSETTA, self).__init__(name)

    def init_ctype(self):
        self.ctype = rosetta()

    @ctypeproperty(float)
    def r_max(): pass

    @ctypeproperty(float)
    def r_lin(): pass

    @ctypeproperty(float)
    def r_sw(): pass

        
class ForcefieldFactory(object):

    ForceField = namedtuple('ForceField', ['cellsize', 'cls'])
    
    forcefields = {'rosetta': ForceField(5.51, ROSETTA),
                   'prolsq': ForceField(3.71, PROLSQ)}
    
    @classmethod
    def create_forcefield(cls, name, universe, n_bins=500):

        if not name.lower() in cls.forcefields:
            raise ValueError('Forcefield {} not supported'.format(name))

        settings = cls.forcefields[name.lower()]
        nblist = NBList(settings.cellsize, n_bins, 500, universe.n_particles)

        forcefield = settings.cls()
        forcefield.nblist = nblist
        forcefield.n_types = 1
        forcefield.types = np.zeros(universe.n_particles,'i')
        forcefield.k = np.ones((forcefield.n_types, forcefield.n_types))
        forcefield.d = np.full((forcefield.n_types, forcefield.n_types),
                               settings.cellsize)

        return forcefield
    
