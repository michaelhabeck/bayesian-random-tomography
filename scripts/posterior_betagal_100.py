"""
Create betagal simulation with 100 particles. 
"""
import numpy as np
import isdbeads as isd

from csb.bio.io import mrc


# settings
n_particles = 100
n_atoms = 32500
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
filename = '../data/betagal_class_averages.mrcs'
reader = mrc.DensityMapReader(filename)
images = reader.read().data.copy()
margin = 20
images = images[margin:-margin, margin:-margin]
pixelsize = 3.54

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
    'bead model',
    likelihoods = likelihoods, 
    priors = [prior]
)

# predict radius of gyration
Rg = 2.74 * (n_atoms/10.)**0.34
rgyr = simulation.create_radius_of_gyration(1., 'exponential')
rgyr.rate = 10. 

# force Rg term to be treated as prior
posterior.priors.append(rgyr)

# spherical initial structure
params['coordinates'].set(isd.random_sphere(n_particles, Rg))
        
posterior.update()
    
