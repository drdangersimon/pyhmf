from dohmf import * 
import numpy as np,numpy.random
np.random.seed(1)
config.nthreads=7
arr,earr = get_data(npix=100,outlfrac=0.001,ncens=2,ndat=1000)                    
eigs=get_firstvec(arr,earr,10)
neweigve, As= get_hmf(arr, earr, eigs.T,nit=130)
neweigve1, As= get_hmf_smooth(arr, earr, neweigve,nit=130)
