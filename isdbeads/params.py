import spin
import numpy as np

from .core import Nominable
from .grid import Grid

from csb.core import validatedproperty, typedproperty
from csb.bio.io import mrc

from collections import OrderedDict


class Parameter(Nominable):
    """Parameter

    Base class for the parameters of a probabilistic model.
    """
    def __init__(self, name):
        """Parameter

        Base class for the parameters of a probabilistic model.

        Parameters
        ----------        
        name : string
            Parameter's name.  
        """
        self.name = name
        self._value = None
        
        self.set_default()
        
    def set_default(self):
        pass

    def get(self):
        if self._value is None:
            msg = 'Parameter not set'
            raise ValueError(msg)
        return self._value

    def set(self, value):
        raise NotImplementedError

    def __str__(self):
        s = super(Parameter, self).__str__()
        v = self._value
        if np.iterable(v):
            v = '{0:.2f}'.format(v[0]) if type(v[0]) == float else v[0]
            v = '[{0},...]'.format(v)
        return s.replace(')',', {})'.format(v))

    
class Location(Parameter):
    """Location

    Scalar location parameter. 
    """
    def set_default(self):
        self.set(0.)

    def set(self, value):
        self._value = float(value)

        
class Scale(Parameter):
    """Scale

    Non-negative scalar factor
    """
    def set_default(self):
        self.set(1.)

    def set(self, value):
        value = float(value)
        if value < 0.:
            msg = '{} must be non-negative'
            raise ValueError(msg.format(self.__class__.__name__))
        self._value = value

        
class Precision(Scale):
    """Precision

    Inverse variance
    """
    pass


class Array(Parameter):

    def __init__(self, name, size):
        super(Array, self).__init__(name)
        self._value = np.ascontiguousarray(np.zeros(int(size)))

    def set(self, value):
        self._value[...] = np.reshape(value, (-1, ))

    def __len__(self):
        return len(self._value)


class Rotation(Array):

    parameterizations = dict(
        euler = spin.EulerAngles,
        quaternion = spin.Quaternion,
        axisangle = spin.AxisAngle,
        expmap = spin.ExponentialMap
    )
    
    def __init__(self, name, parameterization='euler'):
        self._rotation = Rotation.parameterizations[parameterization]()
        super(Rotation, self).__init__(name, self._rotation.dofs.size)
        self._value = self._rotation.dofs
        
    def set(self, value):
        value = np.array(value)
        if value.ndim == 2:
            self._rotation.matrix = value
        else:
            super(Rotation, self).set(value)
            
    @property
    def matrix(self):
        return self._rotation.matrix
        
        
class Coordinates(Array):
    """Coordinates

    Three-dimensional Cartesian coordinates. 
    """
    def __init__(self, universe, name='coordinates'):
        super(Coordinates, self).__init__(name, 3*universe.n_particles)
        self._value = np.ascontiguousarray(universe.coords.reshape(-1,))

        
class Forces(Coordinates):
    """Forces

    Cartesian gradient. 
    """
    def __init__(self, universe, name='forces'):
        super(Forces, self).__init__(universe, name)
        self._value = np.ascontiguousarray(universe.forces.reshape(-1,))

        
class Distances(Array):
    """Distances

    Class for storing and evaluating inter-particle distances. In addition to
    the distances, this class also stores the indices of the particles between
    which the distances are defined. 
    """
    def __init__(self, pairs, name='distances'):
        """Distances

        Pairwise distances between particles

        Parameters
        ----------
        pairs: iterable
            2-tuples specifying the particles whose pairwise distances will be
            computed. 
        """
        super(Distances, self).__init__(name, len(pairs))
        i, j = np.transpose(pairs).astype(np.int32)
        self.first_index = i
        self.second_index = j

    def set(self, distances):
        if np.any(distances < 0.):
            raise ValueError('Distances must be non-negative')
        super(Distances, self).set(distances)
        
    @validatedproperty
    def first_index(values):
        """First element of the index tuple, i.e. if distances are defined
        between `i` and `j`, this array stores all `i`. """
        return np.ascontiguousarray(values)

    @validatedproperty
    def second_index(values):
        """Second element of the index tuple, i.e. if distances are defined
        between `i` and `j`, this array stores all `j`. """
        return np.ascontiguousarray(values)

    @property
    def pairs(self):
        """Generates all index tuples. """
        for i in xrange(len(self)):
            yield (self._first_index[i], self._second_index[i])

            
class Image(Array):
    """ Image

    Image data defined on a regular pixel grid.
    """
    def __init__(self, grid, name='volume'):
        """Image

        Image data defined over a regular pixel grid. 

        Parameters
        ----------
        grid: instance of Grid
            2D cubic grid. 
        """
        self.grid = grid
        super(Image, self).__init__(name, grid.size)

        
    @classmethod
    def from_image(cls, filename):
        """Read volume data from mrcfile and return instance of volume data. """
        data = imread(filename)
        grid = Grid(1., np.zeros(2), data.shape)
        image = cls(grid)
        image.set(data)
        return image

    
    @validatedproperty
    def grid(value):
        if not isinstance(value, Grid):
            raise TypeError('Grid expected')
        elif value.ndim != 2:
            raise ValueError('2D cubic grid expected')
        return value

            
class Volume(Array):
    """ Volume

    Volumetric data defined on a regular voxel grid.
    """
    def __init__(self, grid, name='volume'):
        """Volume

        Volumetric data defined over a regular voxel grid. 

        Parameters
        ----------
        grid: instance of Grid
            3D cubic grid. 
        """
        self.grid = grid
        super(Volume, self).__init__(name, grid.size)

        
    def write_mrc(self, filename):
        """Write volume data to file in MRC format. """
        data = self.get().reshape(self.grid.shape)
        density = mrc.DensityInfo(data, self.grid.spacing, self.grid.origin)
        writer = mrc.DensityMapWriter()
        with open(filename, 'w') as handle:
            writer.write(handle, density)

            
    @classmethod
    def from_mrc(cls, filename):
        """Read volume data from mrcfile and return instance of volume data. """
        reader = mrc.DensityMapReader(filename)
        data = reader.read()
        grid = Grid(data.spacing[0], data.origin, data.data.shape)
        volume = cls(grid)
        volume.set(data.data)
        return volume

    
    @validatedproperty
    def grid(value):
        if not isinstance(value, Grid):
            raise TypeError('Grid expected')
        elif value.ndim != 3:
            raise ValueError('3D cubic grid expected')
        return value

            
class Parameters(object):
    """Parameters

    Class holding all model parameters, data and hyper-parameters. This class
    is shared among all probabilities to make sure that the probabilities
    always use the same parameters.
    """
    def __init__(self):
        self._params = OrderedDict()
        
    def add(self, param):
        if param.name in self._params:
            raise ValueError('Parameter "{}" already added'.format(param))
        self._params[param.name] = param

    def update(self, other_params, ignore_duplications=False):
        for param in other_params:
            try:
                self.add(param)
            except ValueError:
                if not ignore_duplications:
                    raise Exception('Duplicated parameter "{}"'.format(param))

    def __str__(self):
        s = ['Parameters:']
        for param in self:
            s.append('    {}'.format(param))
        return '\n'.join(s)
    
    def __iter__(self):
        return iter(self._params.values())

    def get(self):
        return [param.get() for param in self]

    def set(self, values):
        for param, value in zip(self, values):
            param.set(value)

    def __getitem__(self, name):
        return self._params[name]

    def as_dict(self):
        d = dict()
        for param in self:
            d[param.name] = param.get()
        return d
