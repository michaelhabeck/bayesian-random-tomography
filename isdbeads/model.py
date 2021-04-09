import numpy as np
import scipy.ndimage as ndimage

from matplotlib.pylab import imread

from .utils import image_center
from .grid import Grid

from .params import (
    Parameter,
    Array,
    Location,
    Scale, 
    Distances,
    Image,
    Volume,
    Rotation
)

from .likelihood import (
    Normal, 
    GaussianMixture, 
)

class MockData(Parameter):
    """ MockData

    Theory that allows the computation of idealized data from other parameters.
    """
    def update(self, params):        
        pass

    
class ModelDistances(Distances, MockData):
    """ModelDistances

    Class for storing and *evaluating* inter-particle distances. 
    """
    def __init__(self, pairs, name='distances'):
        """Distances

        Pairwise distances between particles

        Parameters
        ----------
        pairs : iterable
            2-tuples specifying the particles whose pairwise distances will be
            computed. 
        """
        super(ModelDistances, self).__init__(pairs, name)

    def update(self, params):
        from .distance import calc_data
        calc_data(
            params['coordinates'].get(),
            self.first_index,
            self.second_index,
            self._value
        )

    def update_forces(self, derivatives, params):
        """Computes the Cartesian gradient assuming that an instance of
        'Distances' is passed as the dataset. 
        """
        from .distance import update_forces
        update_forces(
            params['coordinates'].get(),
            self.first_index,
            self.second_index,
            self._value,
            derivatives,
            params['forces'].get()
        )

        
class ModelImage(Image, MockData):
    """ModelImage

    Class for storing and *evaluating* 2D images. 
    """
    fourier_settings = dict(
        n_spacing = 2, 
        n_neighbors = 4,
        filter_settings = dict(mode='constant', cval=0.0)
    )
    
    def __init__(self, grid, sigma=1., name='image', params=None,
                 rotation_params='euler'):
        """Image

        Image data defined over a regular pixel grid. 

        Parameters
        ----------
        grid: instance of Grid
            2D cubic Grid. 
        """
        super(ModelImage, self).__init__(grid, name)
        self.fourier = False
        self.sigma = float(sigma)
        self.mode = 0
        self._background = Location(self.name + '.background')
        self._scale = Scale(self.name + '.scale')
        self._rotation = Rotation(self.name + '.rotation', rotation_params)
        self._translation = Array(self.name + '.translation', 2)

        self.rotation = np.eye(3)
        self.translation = np.zeros(2)
        
        # TODO: make parameter?
        nsigma = 5
        self.n_neighbors = int(np.ceil(nsigma*sigma / grid.spacing))

        if params:
            params.add(self._background)
            params.add(self._scale)

            
    def _calc_image(self, coords, weights=None, mode=0):

        from . import image

        weights = np.ones(len(coords)) if (weights is None) else weights
        args = (
            coords,
            self.sigma,
            weights,
            self.grid.origin,
            np.array(self.grid.shape, dtype=np.int32),
            self.grid.spacing,
            self.n_neighbors,
            self._value
        )

        self.set(0.)
        if mode == 0:
            image.calc_image(*args)
        elif mode == 1:
            image.calc_image_fast(*args)
        elif mode == 2:
            image.calc_image_fast2(*args)
        else:
            raise ValueError('Mode "{}" unknown'.format(mode))

        
    def _calc_forces(self, derivatives, coords, forces, weights=None, mode=0):

        from . import image

        weights = np.ones(len(coords)) if (weights is None) else weights
        args = (
            coords, 
            self.sigma,
            weights,
            self.grid.origin,
            np.array(self.grid.shape, dtype=np.int32), 
            self.grid.spacing,
            self.n_neighbors, 
            derivatives,
            forces
        )
        
        if mode == 0:
            image.calc_forces(*args)
        elif mode == 1:
            image.calc_forces_fast(*args)
        elif mode == 2:
            image.calc_forces_fast2(*args)
        else:
            raise ValueError('Mode "{}" unknown'.format(mode))

        
    def calc_image(self, coords, weights=None):

        if not self.fourier:
            self._calc_image(coords, weights, self.mode)
            self._value *= self.scale
            self._value += self.background
            return
        
        filter_settings = self.fourier_settings['filter_settings']
        original_params = self.sigma, self.n_neighbors

        n_neighbors = self.fourier_settings['n_neighbors']
        n_spacing = self.fourier_settings['n_spacing']
        sigma = self.grid.spacing / n_spacing

        # compute high-res map
        self.sigma, self.n_neighbors = sigma, n_neighbors
        self._calc_image(coords, weights, self.mode)
        self.sigma, self.n_neighbors = original_params

        # blur map with Gaussian filter
        h = np.sqrt(self.sigma**2 - sigma**2) / self.grid.spacing
        y = ndimage.gaussian_filter(
            self._value.reshape(self.grid.shape), sigma=h, **filter_settings)
        
        self.set(self.scale * y.flatten() + self.background)

        
    def calc_forces(self, derivatives, coords, forces, weights=None):

        if not self.fourier:
            return self._calc_forces(
                self.scale * derivatives, coords, forces, weights, self.mode
            )

        filter_settings = self.fourier_settings['filter_settings']
        original_params = self.sigma, self.n_neighbors

        n_neighbors = self.fourier_settings['n_neighbors']
        n_spacing = self.fourier_settings['n_spacing']
        sigma = self.grid.spacing / n_spacing

        # blur derivatives with Gaussian filter
        h = np.sqrt(self.sigma**2 - sigma**2) / self.grid.spacing
        g = ndimage.gaussian_filter(
            derivatives.reshape(self.grid.shape), sigma=h, **filter_settings)

        # map blurred derivatives by applying the chain rule
        self.sigma, self._n_neighbors = sigma, n_neighbors
        self._calc_forces(
            self.scale * g.flatten(), coords, forces, weights, self.mode
        )
        self.sigma, self._n_neighbors = original_params

        
    def update(self, params):
        coords = params['coordinates'].get().reshape(-1, 3)
        R, t = self._rotation.matrix, self.translation
        self.calc_image(coords.dot(R[:2].T) + t)

        
    def update_forces(self, derivatives, params):
        """Computes the Cartesian gradient. """
        coords = params['coordinates'].get().reshape(-1, 3)
        forces = np.zeros((len(coords), 2))
        R, t = self._rotation.matrix[:2], self.translation
        
        self.calc_forces(
            derivatives, 
            coords.dot(R.T) + t, 
            forces
        )
        params['forces']._value += forces.dot(R).flatten()

    @property
    def scale(self):
        """Scaling factor. """
        return self._scale.get()

    @scale.setter
    def scale(self, value):
        self._scale.set(value)

    @property
    def background(self):
        """Constant background / offset. """
        return self._background.get()

    @background.setter
    def background(self, value):
        self._background.set(value)

    @property
    def translation(self):
        return self._translation.get()

    @translation.setter
    def translation(self, value):
        self._translation.set(value)

    @property
    def rotation(self):
        return self._rotation.get()

    @rotation.setter
    def rotation(self, value):
        self._rotation.set(value)
        
        
class ModelVolume(Volume, MockData):
    """ModelVolume

    Class for storing and *evaluating* 3D density maps. 
    """
    fourier_settings = dict(
        n_spacing = 2, 
        n_neighbors = 4,
        filter_settings = dict(mode='constant', cval=0.0)
    )
    
    def __init__(self, grid, sigma=1., name='volume', params=None):
        """Volume

        Volumetric data defined over a regular voxel grid. 

        Parameters
        ----------
        grid: instance of Grid
            3D cubic Grid. 
        """
        super(ModelVolume, self).__init__(grid, name)
        self.fourier = False
        self.sigma = float(sigma)
        self.mode = 0
        self._background = Location(self.name + '.background')
        self._scale = Scale(self.name + '.scale')

        # TODO: make parameter?
        nsigma = 5
        self.n_neighbors = int(np.ceil(nsigma*sigma / grid.spacing))

        if params:
            params.add(self._background)
            params.add(self._scale)
        
    def _calc_map(self, coords, weights=None, mode=0):

        from . import volume

        weights = np.ones(len(coords)) if (weights is None) else weights
        args = (
            coords,
            self.sigma,
            weights,
            self.grid.origin,
            np.array(self.grid.shape, dtype=np.int32),
            self.grid.spacing,
            self.n_neighbors,
            self._value
        )

        self.set(0.)
        if mode == 0:
            volume.calc_map(*args)
        elif mode == 1:
            volume.calc_map_fast(*args)
        elif mode == 2:
            volume.calc_map_fast2(*args)
        else:
            raise ValueError('Mode "{}" unknown'.format(mode))

        
    def _calc_forces(self, derivatives, coords, forces, weights=None, mode=0):

        from . import volume

        weights = np.ones(len(coords)) if (weights is None) else weights
        # TODO: check if necessary
        # self.set(0.)

        args = (
            coords, 
            self.sigma,
            weights,
            self.grid.origin,
            np.array(self.grid.shape, dtype=np.int32), 
            self.grid.spacing,
            self.n_neighbors, 
            derivatives,
            forces
        )
        
        if mode == 0:
            volume.calc_forces(*args)
        elif mode == 1:
            volume.calc_forces_fast(*args)
        elif mode == 2:
            volume.calc_forces_fast2(*args)
        else:
            raise ValueError('Mode "{}" unknown'.format(mode))

        
    def calc_map(self, coords, weights=None):

        if not self.fourier:
            self._calc_map(coords, weights, self.mode)
            self._value *= self.scale
            self._value += self.background
            return
        
        filter_settings = self.fourier_settings['filter_settings']
        original_params = self.sigma, self.n_neighbors

        n_neighbors = self.fourier_settings['n_neighbors']
        n_spacing = self.fourier_settings['n_spacing']
        sigma = self.grid.spacing / n_spacing

        # compute high-res map
        self.sigma, self.n_neighbors = sigma, n_neighbors
        self._calc_map(coords, weights, self.mode)
        self.sigma, self.n_neighbors = original_params

        # blur map with Gaussian filter
        h = np.sqrt(self.sigma**2 - sigma**2) / self.grid.spacing
        y = ndimage.gaussian_filter(
            self._value.reshape(self.grid.shape), sigma=h, **filter_settings)
        
        self.set(self.scale * y.flatten() + self.background)

        
    def calc_forces(self, derivatives, coords, forces, weights=None):

        if not self.fourier:
            return self._calc_forces(
                self.scale * derivatives, coords, forces, weights, self.mode
            )

        filter_settings = self.fourier_settings['filter_settings']
        original_params = self.sigma, self.n_neighbors

        n_neighbors = self.fourier_settings['n_neighbors']
        n_spacing = self.fourier_settings['n_spacing']
        sigma = self.grid.spacing / n_spacing

        # blur derivatives with Gaussian filter
        h = np.sqrt(self.sigma**2 - sigma**2) / self.grid.spacing
        g = ndimage.gaussian_filter(
            derivatives.reshape(self.grid.shape), sigma=h, **filter_settings)

        # map blurred derivatives by applying the chain rule
        self.sigma, self._n_neighbors = sigma, n_neighbors
        self._calc_forces(
            self.scale * g.flatten(), coords, forces, weights, self.mode
        )
        self.sigma, self._n_neighbors = original_params

        
    def update(self, params):
        self.calc_map(params['coordinates'].get().reshape(-1, 3))

        
    def update_forces(self, derivatives, params):
        """Computes the Cartesian gradient. """
        self.calc_forces(
            derivatives, 
            params['coordinates'].get().reshape(-1, 3),
            params['forces'].get().reshape(-1, 3), 
        )

    @property
    def scale(self):
        """Scaling factor. """
        return self._scale.get()

    @scale.setter
    def scale(self, value):
        self._scale.set(value)

    @property
    def background(self):
        """Constant background / offset. """
        return self._background.get()

    @background.setter
    def background(self, value):
        self._background.set(value)

    
class ProjectedCloud(Array, MockData):
    """ProjectedCloud

    Class for storing and *evaluating* 2D point clouds from a collection of 3D
    points. 
    """    
    def __init__(self, params, name='cloud', rotation_params='quaternion'):
        """ProjectedCloud. """
        n_particles = params['coordinates'].get().size // 3
        super(ProjectedCloud, self).__init__(name, n_particles * 2)
        
        self._rotation = Rotation(self.name + '.rotation', rotation_params)
        self._translation = Array(self.name + '.translation', 2)

        self.rotation = np.eye(3)
        self.translation = np.zeros(2)
        
        params.add(self._rotation)
        params.add(self._translation)

            
    def update(self, params):
        coords = params['coordinates'].get().reshape(-1, 3)
        R, t = self._rotation.matrix, self.translation
        self.set(coords.dot(R[:2].T) + t)

        
    def update_forces(self, derivatives, params):
        """Computes the Cartesian gradient. """
        R = self._rotation.matrix[:2]
        params['forces']._value += np.reshape(
            derivatives, (-1, 2)
        ).dot(R).flatten()
                    

    @property
    def translation(self):
        return self._translation.get()

    @translation.setter
    def translation(self, value):
        self._translation.set(value)

    @property
    def rotation(self):
        return self._rotation.get()

    @rotation.setter
    def rotation(self, value):
        self._rotation.set(value)
        
        
class RadiusOfGyration(MockData):

    def __init__(self, name='rog'):
        """RadiusOfGyration

        Mean distance from center of mass
        """
        super(RadiusOfGyration, self).__init__(name)
        
    def set_default(self):
        self.set(0.)

    def set(self, value):
        value = float(value)
        if value < 0.:
            msg = '{} must be non-negative'
            raise ValueError(msg.format(self.__class__.__name__))
        self._value = value
        
    def update(self, params):
        coords = params['coordinates'].get().reshape(-1, 3)
        Rg = np.mean(np.sum(np.square(coords - coords.mean(0)), 1))**0.5
        self.set(Rg)

    def update_forces(self, derivatives, params):
        coords = params['coordinates'].get().reshape(-1, 3)
        grad = derivatives[0] * (coords - coords.mean(0)) \
             / (len(coords) * self.get())
        forces = params['forces'].get()
        forces += grad.reshape(forces.shape)

        
class ModelFactory:

    @classmethod
    def create_image(
        cls,
        image,
        params,
        pixelsize=1., 
        grid=None,
        rotation=np.eye(3),
        sigma=1.,
        name='image',
        fourier= False, 
        rotation_params='quaternion'
    ):

        image = image.astype(np.float64)

        if grid is None:
            origin = -image_center(image) * pixelsize
            origin = -0.5 * np.array(image.shape) * pixelsize
            grid = Grid(pixelsize, origin, image.shape)

        mock = ModelImage(
            grid,
            sigma,
            name = name + '.mock', 
            rotation_params = rotation_params
        )
        mock.rotation = rotation

        likelihood = Normal(name, image.flatten(), mock, params=params)
        
        for param in (mock._scale, mock._background, mock._rotation):
            params.add(param)

        likelihood.mock.fourier = fourier

        return likelihood

    
    @classmethod
    def create_cloud(
        cls,
        points,
        params,
        rotation=np.eye(3),
        sigma=1.,
        name='cloud',
        rotation_params='quaternion'
    ):

        mock = ProjectedCloud(
            params,
            name = name + '.mock', 
            rotation_params = rotation_params
        )
        mock.rotation = rotation

        likelihood = GaussianMixture(
            name, points.flatten(), mock, ndim=2, params=params
        )
        likelihood.tau = 1 / sigma**2

        return likelihood
    
