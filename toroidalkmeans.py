'''
Here we provide a k-means implementation on a two dimensional toroidal 
geometry.

Can also do k-medians (or k-'any suitable measure of central tendency').

This program makes extensive use of the work of Fahim, Salem, Torkey and 
Ramadan.
Fahim A.M., Salem A.M., Torkey F.A., Ramadan M.A.
J Zhejiang Univ SCIENCE A 2006 7(10):1626-1633

This file is part of toroidal k-means.

Toroidal k-means is free software: you can redistribute 
it and/or modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

author: Simon Wilshin
contact: swilshin@gmail.com
date: Jan 2016
'''

from numpy import array,eye,sum,pi,dot,mean,argmin
import numpy as np

# We are working with a two-torus projected from a three-torus, so the metric 
# is not always the identity matrix.
g2T = (1.0/3.0)*array([
                          [2.0,1.0],
                          [1.0,2.0]
                          ])
# Provide the identity though for the example and completeness
g2 = eye(2)


def euclidDistFunc(x0,x1,g=g2T):
    '''
    Computes the Euclidean distance between all of the vectors in 
    x0 (NxM) and all of the vectors in x1 (kxM), using metric g (MxM).
    Returns the distance matrix (kxN)
    '''
    return(sum((x1.repeat(x0.shape[0]).reshape(x1.shape[0],x1.shape[1],x0.shape[0])-x0.transpose())**2.0,1))

def quotient2TorusDistFunc(x0,x1,g=g2T):
    '''
    Computes the Euclidean distance between all of the vectors in 
    x0 (NxM) and all of the vectors in x1 (kxM), using metric g (MxM), 
    modulo the topology of the torus. The distance here being defined as the 
    length of the shortest path between the two points on the torus.
    Returns the distance matrix (kxN).
    '''  
    dists = ((x1.repeat(x0.shape[0]).reshape(x1.shape[0],x1.shape[1],x0.shape[0])-x0.transpose()+pi)%(2.0*pi))-pi
    return(sum(dists*((dot(g,dists))).transpose(1,0,2),1))

def kstep(x,mu,dists,muFunc=mean):
    '''
    Given an NxM set of points x, kxM length means, the distances 
    between the points and the means (kxN) and a function 
    for computing new means, this function splits up x and 
    then finds the new means.
    '''
    # Cluster
    idxClust = argmin(dists,0)
    # Find and return new means
    return(array([muFunc(x[idxClust==i,:]) for i in range(mu.shape[0])]),idxClust)

def torkmeans(x,mu0,distFunc=euclidDistFunc,muFunc=lambda x: mean(x,0),maxIt=2000,verbose=False):
    '''
    Given an NxM set of points x, a kxM length of initial means 
    guesses, mu0, a function for computing the distance between two 
    points and a function for computing the mean of a co-ordinate,
    this function applies a generalised version of the k-means 
    algorithm and finds clusters
    
    '''
    converged = False
    nmu = mu0
    clusts = None
    nClusts = None
    i = 0
    while not converged:
        i += 1
        if verbose:
            print "Starting iteration ", i
        # Compute distances
        dists = distFunc(x,nmu)
        # Compute new stopping criterion
        if clusts is not None and all(nClusts==clusts):
            converged = True
        else:
            clusts = nClusts
        if i > maxIt:
            converged = True
            print "Warning, did not converge"
        # Computes new means
        nmu,nClusts = kstep(x,nmu,dists,muFunc)
    return(nmu,nClusts)

if __name__=='__main__':
  '''
  An example in which we generate two clusters on a torus and compare our 
  torus k-mean function to that provided by scipy.
  '''
  from numpy import hstack,logical_not,linspace
  from numpy.random import seed,randn
  from scipy.cluster.vq import kmeans2
  
  # Make deterministic for easier debugging
  seed(0)
  
  # Cluster centers and scales
  x0 = array([[0.1,0.1],[0.6,0.6]])
  s0 = [0.1,0.1]
  
  # Generate test data
  N = 1000
  D = 2
  x = 2*pi*(((s0*randn(N,D,2)).transpose(0,2,1)+x0)%1.0)
  xt = hstack(x.transpose(1,2,0)).T
  
  # Interface for toroidal kmeans and scipy kmeans2 are sufficiently similar 
  # we can use lambdas to give them the same call signature reducing 
  # repetition, so lets do that
  from numpy import median
  mu,clusts = dict(),dict()
  meths = {
    'toroidal kmedians': lambda xt,x0: torkmeans(
      xt,
      x0,
      distFunc=lambda x0,x1: quotient2TorusDistFunc(x0,x1,g2),
      muFunc=lambda x: median(x,0) # Performs better here with kmedians
    ),
    'toroidal kmeans': lambda xt,x0: torkmeans(
      xt,
      x0,
      distFunc=lambda x0,x1: quotient2TorusDistFunc(x0,x1,g2),
    ),    
    'scipy kmeans2': kmeans2
  }
    
  # Calculate means and clusters
  for k in meths:
    mu[k],clusts[k] = meths[k](xt,x0)
    clusts[k] = array(clusts[k],dtype=bool)
    if sum(clusts[k][N:])<3*N/4: # If the clusters are swapped, unswap them
      clusts[k] = logical_not(clusts[k])
      mu[k] = mu[k][[1,0]]
  
  # Print performace
  for k in meths:
    print "Performance of ",k
    print 100.0*float(sum(logical_not(clusts[k][:N]))+sum(clusts[k][N:]))/(2*N),"\% correct"
  
  # Make plot of resulting clusters
  tks = linspace(0,2*pi,5)
  tkslbl = ['$0$','$\\pi/2$','$\\pi$','$3\\pi/2$','$2\\pi$']
  
  from pylab import (figure,subplot,scatter,xticks,yticks,xlim,ylim,title,
    tight_layout,savefig)
  figure(figsize=(2+(len(meths)+1)*4,5))
  subplot(1,len(meths)+1,1)
  scatter(*x[:,0,:].T,color='b',marker='+',lw=1,s=49,alpha=0.95)
  scatter(*x[:,1,:].T,color='r',marker='x',lw=1,s=49,alpha=0.95)
  xticks(tks,tkslbl)    
  yticks(tks,tkslbl)
  xlim([0,2*pi])
  ylim([0,2*pi])
  title('ground truth')
  for i,k in enumerate(meths.keys()):
    subplot(1,len(meths)+1,i+2)
    scatter(*xt[:N][clusts[k][:N]].T,color='k',marker='o',s=81,alpha=0.7)
    scatter(*xt[N:][logical_not(clusts[k])[N:]].T,color='k',marker='o',s=81,alpha=0.7)
    scatter(*xt[logical_not(clusts[k])].T,color='b',marker='+',lw=1,s=49,alpha=0.95)
    scatter(*xt[clusts[k]].T,color='r',marker='x',lw=1,s=49,alpha=0.95)
    xticks(tks,tkslbl)  
    yticks([],[])
    xlim([0,2*pi])
    ylim([0,2*pi])
    title(k)
  tight_layout()
  savefig('kmeansperformance.png')