from __future__ import print_function

import os
import numpy as np
import isdbeads as isd

from gibbs import GibbsSampler
from csb.io import dump, load


def create_posterior(filename):
    with open(filename) as script:
        exec(script)
    return posterior

n_particles = (100, 200)[0]
isdfile = 'posterior_betagal_{}.py'.format(n_particles)

posterior = create_posterior(isdfile)
diameter = posterior.priors[0].forcefield.d[0, 0]
gibbs = GibbsSampler(posterior, stepsize=1e-2)
samples = isd.Ensemble(posterior.params)

# run Gibbs sampler 
with isd.take_time('Gibbs sampling'):
    while len(samples) < 500:
        with isd.take_time('Gibbs step'):
            next(gibbs)
        samples.update(posterior.params)
        cc = []
        for image in posterior.likelihoods:
            cc.append(isd.crosscorr(image.data, image.mock.get()))
        print(np.round([np.mean(cc), np.min(cc), np.max(cc)], 3) * 100)

if False:
    # look at structure
    viewer = isd.ChainViewer()
    viewer.pymol_settings += (
        'as sphere',
        'set sphere_scale={0:.1f}'.format(diameter/3.4)
    )
    viewer(posterior.params['coordinates'].get().reshape(-1, 3), cleanup=False)

if False:    
    pklfile = os.path.join('../simulations', isdfile.replace('.py', '.pkl'))
    dump(samples, pklfile)

    
