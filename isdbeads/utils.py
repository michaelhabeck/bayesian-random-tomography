import os
import time
import tempfile
import numpy as np

from csb.bio import structure
from csb.bio.utils import radius_of_gyration

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

from .universe import Universe


def crosscorr(a, b):
    """Cross-correlation coefficient between two arbitrary arrays of the same
    size. """
    return np.corrcoef(a.flatten(), b.flatten())[0, 1]


def rdf(coords, bins=100, r_max=None):
    """
    Radial distribution function. 

    Parameters
    ----------
    coords : ndarray
        List of coordinate arrays stored as rows. 
    bins : int or numpy array
        Distance bins. 
    r_max : positive float or None
        Maximum distance. 
    """
    if np.ndim(coords) == 2:
        coords = [coords]

    d = np.concatenate(list(map(pdist, coords)), 0)
    if r_max is not None:
        d = d[d<r_max]
        
    g, bins = np.histogram(d, bins=bins)
    r = 0.5 * (bins[1:] + bins[:-1])
    return r, g/r**2


def image_center(image):
    """Center of mass of a 2D image. """
    grid = np.indices(image.shape)
    return np.array([np.sum(x*image) for x in grid]) / image.sum()


def randomwalk(n_steps, dim=3):
    """Generate a random walk in n-dimensional space by making steps of fixed
    size (unit length) and uniformly chosen direction.

    Parameters
    ----------
    n_steps :
        Length of random walk, i.e. number of steps

    dim: positive integer
        Dimension of embedding space (default: dim=3)
    """
    # generate isotropically distributed unit vectors
    bonds = np.random.randn(int(n_steps), int(dim))
    norms = np.linalg.norm(bonds, axis=1)
    return np.add.accumulate(bonds / norms[:, np.newaxis], axis=0)


def random_sphere(n_particles, Rg):
    """Generate particles within a sphere such that a desired radius of gyration
    is matched. 
    """
    r = np.random.random(n_particles)**(1/3.)
    phi = np.random.uniform(0., 2*np.pi, size=n_particles)
    theta = np.arccos(np.random.uniform(-1., 1., size=n_particles))
    coords = np.transpose([np.cos(phi) * np.sin(theta) * r,
                           np.sin(phi) * np.sin(theta) * r, 
                           np.cos(theta) * r])
    coords -= coords.mean(0)
    coords *= Rg / radius_of_gyration(coords)
    return coords


def create_universe(n_particles=1, diameter=1):
    """Create a universe containing 'n_particles' Particles of given diameter.
    The coordinates of the particles follow a random walk in 3D space.

    Parameters
    ----------
    n_particles : non-negative number
        Number of particles contained in universe

    diameter : non-negative float
        Particle diameter
    """
    universe = Universe(int(n_particles))
    universe.coords[...] = randomwalk(n_particles) * diameter

    return universe


def make_chain(coordinates, sequence=None, chainid='A'):
    """Creates a Chain instance from a coordinate array assuming that these are
    the positions of CA atoms. 
    """
    if sequence is None: sequence = ['ALA'] * len(coordinates)

    residues = []

    for i in range(len(sequence)):
        residue = structure.ProteinResidue(i+1, sequence[i],
                                           sequence_number=i+1)
        atom = structure.Atom(i+1, 'CA', 'C', coordinates[i])
        atom.occupancy = 1.0
        residue.atoms.append(atom)
        residues.append(residue)
        
    return structure.Chain(chainid, residues=residues)


def segment_structure(structure, cutoff):
    """Find connected components. """
    distances = squareform(pdist(structure))
    contacts = distances <= cutoff
    _, labels = connected_components(contacts)
    clusters, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    segments = [structure[labels == cluster] for cluster in clusters[order]]
    return segments


class Viewer(object):
    """Viewer
    
    A low-level viewer that allows one to visualize 3d arrays as molecular
    structures using programs such as pymol or rasmol.
    """
    def __init__(self, cmd, **options):

        import distutils.spawn

        exe = distutils.spawn.find_executable(str(cmd))
        if exe is None:
            msg = 'Executable {} does not exist'
            raise ValueError(msg.format(cmd))

        self._cmd = str(exe)
        self._options = options
        
    @property
    def command(self):
        return self._cmd

    def __str__(self):
        return 'Viewer({})'.format(self._cmd)

    def write_pdb(self, coords, filename):

        if coords.ndim == 2: coords = coords.reshape(1,-1,3)

        ensemble = structure.Ensemble()

        for i, xyz in enumerate(coords,1):

            chain  = make_chain(xyz)
            struct = structure.Structure('')
            struct.chains.append(chain)
            struct.model_id = i
            
            ensemble.models.append(struct)
        
        ensemble.to_pdb(filename)

    def __call__(self, coords, cleanup=True):
        """View 3d coordinates as a cloud of atoms. """
        tmpfile = tempfile.mktemp()

        self.write_pdb(coords, tmpfile)
        
        os.system('{0} {1}'.format(self._cmd, tmpfile))

        time.sleep(1.)

        if cleanup: os.unlink(tmpfile)

            
class ChainViewer(Viewer):
    """ChainViewer

    Specialized viewer for visualizing chain molecules. 
    """
    def __init__(self):

        super(ChainViewer, self).__init__('pymol')

        self.pymol_settings = (
            'set ribbon_trace_atoms=1',
            'set ribbon_radius=0.75000',
            'set cartoon_trace_atoms=1',
            'set spec_reflect=0.00000',
            'set opaque_background, off',
            'bg_color white',
            'as ribbon',
            'util.chainbow()'
        )

    def __call__(self, coords, cleanup=True):
        """Show 3D coordinates as a ribbon. """
        pdbfile = tempfile.mktemp() + '.pdb'
        pmlfile = pdbfile.replace('.pdb','.pml')
        
        self.write_pdb(coords, pdbfile)
        
        pmlscript = ('load {}'.format(pdbfile),
                     'hide') + \
                     self.pymol_settings

        with open(pmlfile, 'w') as f:
            f.write('\n'.join(pmlscript))
        
        os.system('{0} {1} &'.format(self._cmd, pmlfile))

        time.sleep(2.)

        if cleanup:
            os.unlink(pdbfile)
            os.unlink(pmlfile)
    

