from __future__ import print_function

import os
import numpy as np
import isdbeads as isd

from gibbs import GibbsSampler, optimize_rotations
from csb.io import dump, load


def create_posterior(filename):
    with open(filename) as script:
        exec(script)
    return posterior

n_particles = (200,  500)[0]

posteriorfile = 'posterior_ribosome_{0}.py'.format(n_particles)
posterior = create_posterior(posteriorfile)
diameter = posterior.priors[0].forcefield.d[0,0]
gibbs = GibbsSampler(posterior, stepsize=1e-2)
samples = isd.Ensemble(posterior.params)

# Gibbs sampling
n_samples = (100, 200, 500)[-1]
with isd.take_time('Gibbs sampling'):
    while len(samples) < n_samples:
        with isd.take_time('single step'):
            next(gibbs)
        samples.update(posterior.params)
        cc = []
        for image in posterior.likelihoods:
            cc.append(isd.crosscorr(image.data, image.mock.get()))
        print(np.mean(cc), np.min(cc), np.max(cc))

if False:
    viewer = isd.ChainViewer()
    viewer.pymol_settings += (
        'as sphere',
        'set sphere_scale={0:.1f}'.format(diameter/3.4)
    )
    viewer(posterior.params['coordinates'].get().reshape(-1, 3), cleanup=False)

if False:
    pklfile = os.path.join(
        '../simulations', posteriorfile.replace('.py', '.pkl')
    )
    dump(samples, pklfile)

