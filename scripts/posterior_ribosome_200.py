"""
Create ribosome simulation with 200 particles. 
"""
import os
import numpy as np
import isdbeads as isd
import gibbs

from csb.io import load
from csb.bio.io import mrc
from skimage.transform import resize


# settings
n_particles = 200
n_atoms = 124346 + 68763    
diameter = 1.83 * (float(n_atoms)/n_particles)**0.42
k_forcefield = 175. / diameter**4 * 1.6770644663580867
fourier = True
sigma = 10.


# setup posterior
simulation = isd.ChromosomeSimulation(
    n_particles, 
    forcefield = 'prolsq',
    diameter = diameter, 
    k_forcefield = k_forcefield
)

simulation.create_universe()
universe = simulation.universe
    
simulation.create_params()
params = simulation.params
coords = params['coordinates']
    
# images
filename = '../data/pfrib80S_cavgs.mrc'
reader = mrc.DensityMapReader(filename)
margin = 20
images = reader.read().data.copy()[margin:-margin, margin:-margin, :50]
images = images.astype(np.float64)
pixelsize = 2.68

# downsample
images = resize(images, (70, 70, images.shape[2]), mode='constant',
                anti_aliasing=False)
pixelsize *= 2

likelihoods = []
for i in range(images.shape[2]):
    likelihood = isd.ModelFactory.create_image(
        images[:, :, i], 
        params,
        pixelsize = pixelsize, 
        sigma = sigma,
        name = 'image_{}'.format(i),
        fourier = fourier
    )
    likelihoods.append(likelihood)

# create posterior
prior = simulation.create_prior()
posterior = isd.PosteriorCoordinates(
    'ribosome',
    likelihoods = likelihoods, 
    priors = [prior]
)

# predict radius of gyration
Rg = 2.74 * (n_atoms/10.)**0.34

# create prior term
rgyr = simulation.create_radius_of_gyration(1., 'exponential')
rgyr.rate = 10.

# force Rg term to be treated as prior
posterior.priors.append(rgyr)

# spherical initial structure
params['coordinates'].set(isd.random_sphere(n_particles, Rg))
        
posterior.update()
    
