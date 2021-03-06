import numpy as np

from csb.core import validatedproperty


class Particle(object):
    """Particle
    
    Class for representing a particle. Each particle has a unique serial number
    that is used to access the position of the particle.
    """
    _coords = None

    def __init__(self, serial):
        """Particle

        Initialize a particle by setting a serial number.
        """
        self.serial = serial
        
    def __str__(self):

        s = '{0}(serial={1}, coords={2})'
        return s.format(self.__class__.__name__, self.serial,
                        np.round(self.coords,2))

    __repr__ = __str__

    @classmethod
    def set_coords(cls, coords):
        cls._coords = coords
    
    @validatedproperty
    def serial(value):
        """A unique serial number that is used to access the position of the
        particle in a global coordinate array stored in the Universe instance.
        """
        value = int(value)
        if value < 0:
            msg = 'Serial number must be greater than or equal to zero'
            raise ValueError(msg)
        return value

    @property
    def coords(self):
        return Particle._coords[self.serial]
    
    @coords.setter
    def coords(self, values):
        Particle._coords[self.serial,...] = values
    
    
class Universe(object):
    """Universe
    
    Container for all particles that make up a molecular system. Gives access
    to the forces and coordinates of all particles. 
    """
    def __init__(self, n_particles):
        """Universe

        Initialize Universe by specifying the number of particles contained
        in the universe.
        """
        self._coords = np.zeros((n_particles, 3))
        self._forces = np.zeros((n_particles, 3))

        Particle.set_coords(self.coords)

    def __iter__(self):
        for serial in xrange(self.n_particles):
            yield Particle(serial)

    @property
    def n_particles(self):
        """Number of particles in universe. """
        return len(self._coords)

    @property
    def coords(self):
        """Cartesian coordinates of all particles. """
        return self._coords

    @property
    def forces(self):
        """Gradient with respect to Cartesian coordinates. """
        return self._forces

            
