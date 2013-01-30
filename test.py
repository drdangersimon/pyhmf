from dohmf import * 
import numpy as np,numpy.random
np.random.seed(1)
config.nthreads=7
arr, earr = get_data(npix=100, outlfrac=0.001, ncens=2, ndat=1000)
eigs = get_firstvec(arr,earr,10).T
eigs1, proj1= get_hmf(arr, earr, eigs, nit=130)
eigs2, proj2= get_hmf_smooth(arr, earr, eigs1, nit=130)
_,chisqs = project_only(arr,earr, eigs2, getChisq=True)
