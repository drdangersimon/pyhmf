import numpy as np,scipy,numpy.random as nprand

def get_data(ndat=1000,npix=100,ncens=5,xmax=10,err0=0.1):
	xgrid = np.linspace(-xmax,xmax,npix)
	#cens = nprand.uniform(-xmax,xmin,size=ncens)
	cens = np.linspace(-xmax,xmax,ncens)

	cenids = nprand.randint(ncens,size=ndat)
	sig = 1
	arr=[]
	earr=[]

	for i in range(ndat):
		curerr0 = nprand.exponential(err0)
		errs0 = curerr0+ np.zeros(npix)
		#errs0 = np.random.uniform(0,err0,size=npix)
		errs = np.random.normal(0,errs0,size=npix)

		y=scipy.stats.norm.pdf(xgrid,cens[cenids[i]],sig)+errs
		arr.append(y)
		earr.append(errs0)
		
	return np.array(arr),np.array(earr)

def get_pca(arr):
	arrm=np.matrix(arr)
	eigvals, eigvecs =scipy.linalg.eigh(arrm.T*arrm)
	return eigvals,eigvecs
	


def get_hmf(dat,edat,vecs):
	"""
	dat should have the shape Nobs,Npix
	edat the same thing
	vecs should have the shape (npix, ncomp)
	"""
	ncomp = vecs.shape[1]
	ndat = len(dat)
	npix = len(dat[0])
	g0 = np.matrix(vecs)
	#a step
	As = np.matrix(np.zeros((ndat,ncomp)))
	Gs = np.matrix(vecs)

	for i in range(ndat):
		Fi = np.matrix(dat[i]/edat[i]**2)*Gs
		Covi = np.matrix(np.diag(1./edat[i]**2))
		Gi = Gs.T*Covi*Gs
		Ai = scipy.linalg.solve(Gi, Fi.T)
		As[i,:]=Ai.flatten()
	
	#g step 
	for j in range(npix):
		Covj = np.matrix(np.diag(1./edat[:,j]**2))
		Aj = As.T * Covj * As
		Fj = As.T * np.matrix((dat/edat**2)[:,j]).T
		Gj = scipy.linalg.solve(Aj, Fj)
		Gs[j,:]=Gj.flatten()
	return Gs, As

def get_hmf_smooth(dat,edat,vecs,eps=0.01):
	"""
	dat should have the shape Nobs,Npix
	edat the same thing
	vecs should have the shape (npix, ncomp)
	"""
	ncomp = vecs.shape[1]
	ndat = len(dat)
	npix = len(dat[0])
	As = np.matrix(np.zeros((ndat,ncomp)))
	Gs = np.matrix(vecs)

	#a step
	for i in range(ndat):
		Fi = np.matrix(dat[i]/edat[i]**2)*Gs
		Covi = np.matrix(np.diag(1./edat[i]**2))
		Gi = Gs.T*Covi*Gs
		Ai = scipy.linalg.solve(Gi, Fi.T)
		As[i,:]=Ai.flatten()
	
	Gsold = Gs.copy()
	#g step 
	for j in range(npix):
		Covj = np.matrix(np.diag(1./edat[:,j]**2))
		if j>0 and j<(npix-1):
			Aj = As.T * Covj * As + 2* eps * np.identity(ncomp)
			Fj = As.T * np.matrix((dat/edat**2)[:,j]).T + eps * (Gsold[j-1,:] + Gsold[j+1,:]).T
		elif j==0:
			Aj = As.T * Covj * As + eps * np.identity(ncomp)
			Fj = As.T * np.matrix((dat/edat**2)[:,j]).T + eps *  Gsold[1,:].T
		elif j==npix-1:
			Aj = As.T * Covj * As + eps * np.identity(ncomp)
			Fj = As.T * np.matrix((dat/edat**2)[:,j]).T + eps *  Gsold[npix-2,:].T
		Gj = scipy.linalg.solve(Aj, Fj)
		Gs[j,:]=Gj.flatten()
	return Gs, As

def rescaler(Gs, As):
	eigvals, eigvecs =scipy.linalg.eigh(As.T*As)
	return Gs*np.matrix(eigvecs).T,As*np.matrix(eigvecs).T
	
def full_loop():
	arr,earr = get_data.get_data(npix=101)
	eigva,eigve = get_data.get_pca(arr)
	neweigve, As= get_data.get_hmf(arr, earr, eigve[:,-5:])
	