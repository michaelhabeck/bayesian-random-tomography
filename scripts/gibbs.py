"""
Implementation of the Gibbs sampler for random tomography. 
"""
import numpy as np
import isdbeads as isd

from copy import deepcopy
from csb.io import load
from scipy.spatial.distance import cdist


class GibbsState(object):

    def __init__(self, posterior):
        posterior.update()
        self.value = deepcopy(posterior.params.get())
        self.log_prob = posterior.log_prob()


class NuisanceParamsSampler(object):

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def sample(self):

        a = self.likelihood.mock.background
        b = self.likelihood.mock.scale        
        x = (self.likelihood.mock.get() - a) / b
        y = self.likelihood.data

        # design and normal matrix
        F = np.transpose([np.ones_like(x), x])
        A = np.linalg.inv(F.T.dot(F) + 1e-3 * np.eye(2))
        z = F.T.dot(y)

        # params of gamma distribution
        beta = 0.5 * self.likelihood.beta * (y.dot(y) - z.dot(A.dot(z)))
        alpha = 0.5 * self.likelihood.beta * len(x) - 1

        self.likelihood.tau = tau = np.random.gamma(alpha) / beta
        
        # sample scale and background
        sample = np.random.multivariate_normal(
            A.dot(z), A / self.likelihood.beta / tau
        )
        self.likelihood.mock.background = sample[0]
        self.likelihood.mock.scale = abs(sample[1])

    def optimize(self):

        a = self.likelihood.mock.background
        b = self.likelihood.mock.scale        
        x = (self.likelihood.mock.get() - a) / b
        y = self.likelihood.data

        # design and normal matrix
        F = np.transpose([np.ones_like(x), x])
        A = np.linalg.inv(F.T.dot(F) + 1e-3 * np.eye(2))
        z = F.T.dot(y)

        # params of gamma distribution
        beta = 0.5 * (y.dot(y) - z.dot(A.dot(z)))
        alpha = 0.5 * len(x) - 1

        self.likelihood.tau = tau = alpha / beta
        
        map_estimate = A.dot(z)
        self.likelihood.mock.background = map_estimate[0]
        self.likelihood.mock.scale = abs(map_estimate[1])

        
class GibbsSampler(object):

    def set_replica_params(self):

        self.posterior['tsallis'].q = self.q
        for likelihood in self.posterior.likelihoods:
            likelihood.beta = self.beta

    def __init__(self, posterior, n_steps=10, stepsize=1e-3, n_leaps=10,
                 q=1.0, beta=1.0, resample_frequency=0.1):

        self.beta = float(beta)
        self.q = float(q)

        self.resample_frequency = resample_frequency
        
        self.posterior = posterior
        self.parameter = posterior.params
        
        # HMC for particles positions
        hmc = isd.HamiltonianMonteCarlo(posterior, stepsize)
        hmc.leapfrog.n_steps = int(n_leaps)
        hmc.adapt_until = 1e6
        hmc.n_steps = int(n_steps)
        hmc.activate()

        self.samplers = [hmc]

        # rotation samplers
        for likelihood in posterior.likelihoods:
            self.samplers.append(isd.RotationSampler(likelihood))

        self.state = self.create_state()
            
    def __str__(self):
        return 'GibbsSampler: q={0:.3f}, beta={1:.3f}'.format(self.q, self.beta)

    def create_state(self):
        self.set_replica_params()
        return GibbsState(self.posterior)

    
    ## def sample_nuisance_params(self, image):

    ##     # esimate scale and background
    ##     mock_image = (image.mock.get() - image.mock.background) \
    ##                / image.mock.scale
    ##     design_matrix = np.transpose([np.ones_like(mock_image),
    ##                                   mock_image])
    ##     precision_matrix = design_matrix.T.dot(design_matrix)
    ##     covariance = np.linalg.inv(precision_matrix+1e-3*np.eye(2))
    ##     map_estimate = covariance.dot(design_matrix.T.dot(image.data))
            
    ##     # optimization
    ##     image.mock.background = map_estimate[0]
    ##     image.mock.scale = abs(map_estimate[1])
            
    ##     # sampling
    ##     sample = np.random.multivariate_normal(
    ##         map_estimate, covariance / image.beta / image.tau
    ##     )
    ##     image.mock.background = sample[0]
    ##     image.mock.scale = abs(sample[1])
        
    ##     # sample precision
    ##     residuals = image.data - image.mock.scale * mock_image \
    ##               - image.mock.background
    ##     alpha = 0.5 * image.beta * residuals.size
    ##     beta = 0.5 * image.beta * np.dot(residuals, residuals)
            
    ##     # sampling
    ##     image.tau = np.random.gamma(alpha) / beta
        

    ## def optimize_nuisance_params(self, image):

    ##     # esimate scale and background
    ##     mock_image = (image.mock.get() - image.mock.background) \
    ##                / image.mock.scale
    ##     slope, intercept = np.polyfit(mock_image, image.data, 1)
    ##     image.mock.background = intercept
    ##     image.mock.scale = abs(slope)
            
    ##     # estimate precision
    ##     residuals = image.data - image.mock.scale * mock_image \
    ##               - image.mock.background
    ##     image.tau = 1 / np.var(residuals)

    def sample_nuisance_params(self, image):
        sampler = NuisanceParamsSampler(image)
        sampler.sample()
        
    def optimize_nuisance_params(self, image):
        sampler = NuisanceParamsSampler(image)
        sampler.optimize()
        
    def next(self):

        self.posterior.params.set(self.state.value)
        self.set_replica_params()
        
        # coordinates
        hmc = self.samplers[0]
        hmc.state = hmc.create_state()
        next(hmc)

        # sample projection directions and nuisance parameters
        for image, sampler in zip(self.posterior.likelihoods,
                                  self.samplers[1:]):
            state = sampler.sample(10, self.resample_frequency)[-1]
            sampler.parameter.set(state.value)
            image.update()

            self.sample_nuisance_params(image)
            
        self.state = self.create_state()
            
        return self.state

    
    def sample(self, n_steps=1):
        for i in range(n_steps):
            next(self)
        return self.state

    
    @property
    def stepsize(self):
        return self.samplers[0].stepsize

    @stepsize.setter
    def stepsize(self, value):
        self.samplers[0].stepsize = value

    @property
    def history(self):
        return self.samplers[0].history

    
class GibbsSamplerPoints(GibbsSampler):

    def sample_nuisance_params(self, likelihood):
        p = np.exp(likelihood.log_prob_array())
        y = likelihood.data.reshape(-1, 2)
        x = likelihood.mock.get().reshape(-1, 2)
        d = cdist(y, x, metric='sqeuclidean')

        alpha = 0.5 * likelihood.ndim * likelihood.beta * len(y)
        beta = 0.5 * likelihood.beta * np.sum(p * d)

        likelihood.tau = np.random.gamma(alpha) / beta

        
def set_initial(posterior, pklfile, burnin=-10, max_distance=80):
    """Prepare input for refinement. """
    samples = load(pklfile)
    coords = samples['coordinates'].reshape(len(samples), -1, 3)[burnin:]
    coords = coords.reshape(-1, 3)
    norm = np.linalg.norm(coords, axis=1)
    coords = coords[norm < max_distance]

    n_particles = len(posterior.params['coordinates'].get()) // 3
    
    new_coords = coords[-n_particles:].copy()
    new_coords += np.random.randn(*new_coords.shape)
        
    posterior.params['coordinates'].set(new_coords)
    posterior.update()
    
    cc = []
    for image in posterior.likelihoods:
        for param in (
            image.mock._rotation,
            image.mock._scale,
            image.mock._background
        ):
            param.set(samples[param.name][-1])
            image.update()
            image.precision.set(1/np.var(image.data-image.mock.get()))
            cc.append(isd.crosscorr(image.data, image.mock.get()))
            
    print(np.min(cc), np.max(cc), np.mean(cc))

    
def set_params(posterior, ensemble, index=-1):

    for param in posterior.params:
        if param.name in ('coordinates', 'forces'):
            continue
        if not param.name in ensemble.keys():
            continue
        param.set(ensemble[param.name][index])

    n_particles = len(posterior.params['coordinates'].get()) // 3
    diameter = posterior.priors[0].forcefield.d[0, 0]
    coords = ensemble['coordinates'].reshape(len(ensemble), -1, 3)
    counter = -1
    structures = []
    while np.sum(map(len, structures)) < n_particles:
        segments = isd.segment_structure(coords[counter], 2 * diameter)
        structures.append(segments[0])
        counter -= 1 

    coords = np.vstack(structures)
    posterior.params['coordinates'].set(coords[:n_particles])
    posterior.update()


def optimize_rotations(likelihoods, level=1):
    """Optimize all rotations. """
    quaternions = np.load('../data/quaternions.npz')['level{}'.format(level)]

    model = likelihoods[0].mock
    params = likelihoods[0].params
    data = np.array([likelihood.data for likelihood in likelihoods])
    data -= data.mean(1)[:, np.newaxis]
    data /= data.std(1)[:, np.newaxis]

    cc = []
    for q in quaternions:
        model.rotation = q
        model.update(params)
        image = model.get().copy()
        image -= image.mean()
        image /= image.std()
        cc.append(data.dot(image) / len(image))
    j = np.argmax(cc, axis=0)

    return quaternions[j]

