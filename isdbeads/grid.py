import functools
import numpy as np
import scipy.ndimage as ndimage


class Grid(object):
    """Grid

    Regular cubic grid
    """
    # factory methods
    @classmethod
    def from_bbox(self, bbox, spacing):
        """Create grid from bounding box. """
        shape = np.ceil((bbox[1]-bbox[0])/spacing).astype(int) + 1
        return Grid(spacing, bbox[0], shape)

    
    @classmethod
    def from_points(self, points, spacing, margin=0):
        """Create grid from points. """
        shape = np.ceil(np.ptp(points, 0)/spacing).astype(int) + 1
        origin = np.min(points, 0) - 0.5*margin*spacing
        return Grid(spacing, origin, shape+margin)

    
    def __init__(self, spacing, origin, shape):
        """
        Parameters
        ----------
        spacing : positive float
            Grid spacing.
          
        origin : rank-1 numpy array
            Lower left corner of the cubic grid.

        shape : rank-1 numpy array or list or tuple
            Extent of grid (number of cells in each dimension). 
        """
        self._spacing = float(spacing)
        self._origin = np.array(origin)
        self._shape = np.array(shape)
        
        assert self.spacing > 0
        assert self.origin.ndim == 1
        assert self.origin.shape == self._shape.shape
        assert np.all(self._shape)

        
    def multi_index(self, points, return_remainder=False, rounding=True):
        """Multi-index of nearest lattice points. """
        ufunc = np.round if rounding else np.floor
        ind = ufunc((points-self.origin)/self.spacing).astype(int)
        return ind if not return_remainder else (
            ind, points-self.spacing*ind-self.origin)

    
    def index(self, points, return_remainder=False, rounding=False):
        """Flat index of nearest lattice points. """
        if return_remainder:
            ind, diff = self.multi_index(points, True, rounding)
        else:
            ind = self.multi_index(points, False, rounding)
        ind = np.ravel_multi_index(ind.T, self.shape)
        return ind if not return_remainder else (ind, diff)

    
    def discretize(self, points, weights=None):
        """Map points to grid. """
        ind = self.index(points, rounding=True)
        occ = np.zeros(np.prod(self.shape))

        if weights is None:
            ind, counts = np.unique(ind, return_counts=True)
            occ[ind] = counts
        else:
            ind, labels = np.unique(ind, return_inverse=True)
            occ[ind] = ndimage.sum(weights, labels, index=np.arange(len(ind)))

        return occ.reshape(self.shape)

    
    def coords(self, index):
        """Coordinates of cells given by index. """
        cells = np.transpose(np.unravel_index(index, self.shape))
        return self.origin + self.spacing * cells

    
    def inside(self, points):
        """Return mask indicating which points lie inside the grid. """
        return np.logical_and.reduce(points >= self.bbox[0], 1) \
             & np.logical_and.reduce(points <= self.bbox[1], 1)

    
    @property
    def spacing(self):
        """Grid spacing. """
        return self._spacing

    
    @property
    def origin(self):
        """Lower left corner of the grid. """
        return self._origin

    
    @property
    def shape(self):
        """Number of cells in each dimension. """
        return tuple(self._shape)

    @property
    def size(self):
        return np.prod(self._shape)
    
#    @property
#    @functools.lru_cache(1)
    @property
    def bbox(self):
        """Bounding box. """
        return self.origin, self.origin + self.spacing*(self._shape-1)

    
    @property
    def ndim(self):
        """Dimensionality of embedding space. """
        return len(self.origin)

    
    @property
    def axes(self):
        return tuple(origin + self.spacing * np.arange(ncells)
                     for origin, ncells in zip(self.origin, self._shape))

    
    @property
    def lattice(self):
        """Lattice points spanning the grid. """
        axes = self.axes
        lattice = np.array(np.meshgrid(*axes, indexing='ij'))
        lattice = np.transpose(lattice, range(1, len(axes)+1) + [0])
        return lattice.reshape(-1, self.ndim)

    
