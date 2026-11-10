"""Shared spectral-bin amplitudes on the unchanged native objective grid."""
import numpy as np
from scipy.optimize import nnls
from .engine import Limit


def _coordinate_solve(gram,rhs,channels,widths,*,nonnegative,active=None):
    """Block-coordinate solution using channel sufficient statistics only.

    No observation-sized design matrix is allocated. Components can use
    different spectral resolutions; active NNLS groups define tangent space.
    """
    count,n=rhs.shape[:2]
    tail=rhs.shape[2:]
    values=np.zeros_like(rhs)
    maps=[np.unique(channels//width,return_inverse=True)[1] for width in widths]
    groups=[np.bincount(m).size for m in maps]
    converged=False
    for iteration in range(160):
        before=values.copy()
        for j,(labels,ng) in enumerate(zip(maps,groups)):
            remainder=rhs[:,j]-np.einsum('ck,ck...->c...',gram[:,j,:],values)+gram[:,j,j].reshape((-1,)+(1,)*len(tail))*values[:,j]
            sums=np.zeros((ng,)+tail)
            np.add.at(sums,labels,remainder)
            den=np.bincount(labels,weights=gram[:,j,j],minlength=ng)
            out=np.divide(sums,den.reshape((-1,)+(1,)*len(tail)),out=np.zeros_like(sums),
                          where=(den>1e-100).reshape((-1,)+(1,)*len(tail)))
            if nonnegative:out=np.maximum(out,0.)
            if active is not None:
                allowed=np.bincount(labels,weights=active[:,j].astype(float),minlength=ng)>0
                out[~allowed]=0.
            values[:,j]=out[labels]
        if np.max(abs(values-before))<=1e-8*(1+np.max(abs(values))):
            converged=True;break
    return values,converged,iteration+1


def mixed_evaluate(obj,x,n,scatter,derivative=False,keep=False):
    from .scintillation import shapes
    d=obj.d;channels=obj.channels;widths=obj.spectral_group_sizes
    if len(widths)!=n or min(widths)<1:raise ValueError('Invalid per-component spectral bins')
    d.e.budget['evaluations']+=1
    d.e.budget['native_work']+=len(channels)*len(obj.t)*n*(4 if derivative else 1)
    if (d.e.budget['evaluations']>d.c['max_evaluations'] or
            d.e.budget['native_work']>d.c['max_native_work']):
        raise Limit('mixed spectral fit exceeded shared work budget')
    gram=np.zeros((len(channels),n,n));rhs=np.zeros((len(channels),n));constant=0.
    size=2*n+1+int(scatter)
    tile=max(1,min(16,d.c['tile_derivative_cells']//max(1,len(obj.t)*size)))
    for start in range(0,len(channels),tile):
        ii=channels[start:start+tile]
        y=d.e.raw[np.ix_(ii,obj.train)].astype(float);y-=y.mean(axis=1,keepdims=True)
        h,_=shapes(x,n,scatter,d.e.coords,ii,obj.t)
        h-=h.mean(axis=1,keepdims=True)
        var=d.var[ii]
        gram[start:start+len(ii)]=np.einsum('ctj,ctk,c->cjk',h,h,1/var)
        rhs[start:start+len(ii)]=np.einsum('ctj,ct,c->cj',h,y,1/var)
        constant+=float(np.sum(y*y/var[:,None]))
    a,converged,_=_coordinate_solve(gram,rhs,channels,widths,nonnegative=True)
    if not converged:raise Limit('mixed NNLS did not converge within 160 coordinate sweeps')
    loss=constant-2*np.sum(rhs*a)+np.einsum('cj,cjk,ck->',a,gram,a)
    g=np.zeros(size);curvature=np.zeros((size,size))
    if derivative:
        cross=np.zeros((len(channels),n,size))
        for start in range(0,len(channels),tile):
            ii=channels[start:start+tile];aa=a[start:start+len(ii)]
            y=d.e.raw[np.ix_(ii,obj.train)].astype(float);y-=y.mean(axis=1,keepdims=True)
            h,derivs=shapes(x,n,scatter,d.e.coords,ii,obj.t,True)
            h-=h.mean(axis=1,keepdims=True)
            r=y-np.einsum('ctj,cj->ct',h,aa)
            jac=np.zeros((size,len(ii),len(obj.t)))
            for j,dj in enumerate(derivs):
                dj-=dj.mean(axis=2,keepdims=True)
                jac[2*j:2*j+2]=dj[:2]*aa[None,:,j,None]
                jac[2*n]+=dj[2]*aa[:,j,None]
                if scatter:jac[-1]+=dj[3]*aa[:,j,None]
            var=d.var[ii]
            g+=np.einsum('pct,ct,c->p',jac,r,1/var)
            curvature+=np.einsum('pct,qct,c->pq',jac,jac,1/var)
            cross[start:start+len(ii)]=np.einsum('ctj,pct,c->cjp',h,jac,1/var)
        projection,projected,_=_coordinate_solve(gram,cross,channels,widths,
                                                 nonnegative=False,active=a>0)
        if projected:
            correction=np.einsum('cjp,cjq->pq',cross,projection)
            curvature-=(correction+correction.T)/2
            # Roundoff at almost collinear components can make a tiny negative
            # eigenvalue. Use a positive-semidefinite GN approximation only.
            eigen,vectors=np.linalg.eigh((curvature+curvature.T)/2)
            curvature=(vectors*np.maximum(eigen,0))@vectors.T
        # Otherwise retain the positive unprojected curvature, while the
        # envelope gradient remains valid. Line search checks actual loss.
    return float(max(0.,loss)),g,curvature,a if keep else None


def pooled_evaluate(obj, x, n, scatter, derivative=False, keep=False):
    from .scintillation import shapes
    d = obj.d
    d.e.budget['evaluations'] += 1
    d.e.budget['native_work'] += len(obj.channels)*len(obj.t)*max(1,n)*(3 if derivative else 1)
    if (d.e.budget['evaluations'] > d.c['max_evaluations'] or
            d.e.budget['native_work'] > d.c['max_native_work']):
        raise Limit('combined spectral fit exceeded shared work budget')
    size = 2*n+1+int(scatter)
    grad, gram = np.zeros(size), np.zeros((size,size))
    loss, amps = 0., []
    labels = obj.channels//obj.spectral_group_size
    groups = np.split(obj.channels,np.flatnonzero(np.diff(labels))+1)
    for ii in groups:
        y = d.e.raw[np.ix_(ii,obj.train)].astype(float)
        y -= y.mean(axis=1,keepdims=True)
        h, derivs = shapes(x,n,scatter,d.e.coords,ii,obj.t,derivative)
        h -= h.mean(axis=1,keepdims=True)
        weight = 1/np.sqrt(d.var[ii,None])
        design = (h*weight[:,:,None]).reshape(-1,n)
        norms = np.linalg.norm(design,axis=0)
        active = norms > 1e-100
        a = np.zeros(n)
        if active.any():
            a[active] = nnls(design[:,active]/norms[active],(y*weight).ravel(),
                             maxiter=max(100,10*n))[0]/norms[active]
        residual = (y-np.einsum('ctn,n->ct',h,a))*weight
        loss += float(np.sum(residual**2))
        if keep: amps.append(np.repeat(a[None,:],len(ii),axis=0))
        if derivative:
            jac = np.zeros((size,len(ii),len(obj.t)))
            for j, dj in enumerate(derivs):
                dj -= dj.mean(axis=2,keepdims=True)
                jac[2*j:2*j+2] = dj[:2]*a[j]
                jac[2*n] += dj[2]*a[j]
                if scatter:jac[-1] += dj[3]*a[j]
            jw = (jac*weight[None,:,:]).reshape(size,-1)
            active = a > 0
            if active.any():
                basis = design[:,active]
                jw -= (basis@np.linalg.lstsq(basis,jw.T,rcond=1e-10)[0]).T
            grad += jw@residual.ravel()
            gram += jw@jw.T
    return loss,grad,gram,np.concatenate(amps) if keep else None
