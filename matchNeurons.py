#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Match two or more recordings using the point locations found in heatData.mat.
This script uses pyCPD to match two pointsets and outputs a correspondence list of coordinates.
Should perform well on immobillized animals where clustering is unneccessary.
Needs the prediction code repository installed.
a) clone PredictionCode repository from my github
b) cd PredictionCode/
c) python setup.py install --user
@author: monika
"""
import os
import scipy.io
import numpy as np
import matplotlib.pylab as plt
from  sklearn.metrics.pairwise import pairwise_distances
from prediction.pycpd import deformable_registration, rigid_registration
################################################
#
#function for aligning two sets of data
#
################################################
def registerWorms(R, X, dim=3, nr = True):
    """use pycpd code to register two worms. R is the reference. """
    if nr:
        registration = rigid_registration
        reg = registration(R[:,:dim],X, tolerance=1e-5)
        reg.register(callback=None)
        Xreg = reg.TY
    else:
        Xreg = X
    registration = deformable_registration
    reg = registration(R[:,:dim],Xreg, tolerance=1e-8)
    reg.register(callback=None)
    Xreg = reg.TY
    return Xreg

def assignCorrespondences(R, Xreg, maxD = 1):
    """Use pairwise distances between reference R and registered pointset Xreg to assign matching IDs.
        if a cutoff distance is given, matches with a larger distance will remain unmatched.
    """
    #uniquely assign IDs
    D = pairwise_distances(R, Xreg)
    tmpID = np.ones(len(R))*-1
    for i in range(len(R)):
        # check if we are larger than a cutoff distance
        if D.min() < maxD:
            # find best match, delete that option
            ym, xm = np.unravel_index(D.argmin(), D.shape)
            D[ym,:] = np.inf
            D[:,xm] = np.inf
            tmpID[ym] = xm
    return tmpID
    
def main(show = 1):
    
    # load files. First one is the reference. All others will be aligned w.r.t that one
    # define and output file
    outfile = '/tigress/LEIFER/Monika/out.txt'
    path = '/tigress/LEIFER/PanNeuronal/'
    file1 = '20190509/BrainScanner20190509_150032'
    file2 = '20190511/BrainScanner20190511_173700'
    file3 = '20190514/BrainScanner20190514_164647'
    files = [os.path.join(path, fn, 'heatData.mat') for fn in [file1, file2, file3]]
    ventral = [-1, 1, -1]
    # load pointset in heatData
    PSets = []
    for fname, v in zip(files, ventral):
        X = scipy.io.loadmat(fname)['XYZcoord']
        # Correct ventral alignment - ventral coord should be up top
        X[:,1] *= v
        # z-score
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)
        print "Worm with {} Neurons loaded.".format(len(X))
        PSets.append(X)
    
    
    # align as many files as you want sequentially
    # Define the global reference
    R = PSets[0]
    PSets_reg = []
    ids = np.ones((len(R), len(PSets)-1))*np.nan
    for index, X in enumerate(PSets[1:]):
        Xreg = registerWorms(R, X, dim=3, nr=1)
        PSets_reg.append(Xreg)
        ids[:,index] = assignCorrespondences(R, Xreg, maxD = 0.5)
        print '{} Neurons matched with reference.'.format(np.sum(ids[:,index]>0))
    ids = np.array(ids)#, dtype = int)
    np.savetxt(outfile, ids,fmt='%i', header = "".join(files))
    s = 30
    
    if show:
        plt.figure('Input datasets')
        for ci, C in enumerate(PSets):
            plt.subplot(len(PSets),1,ci+1)
            if ci==0:
                plt.title('Reference')
            plt.scatter(C[:,0], C[:,1], s = 30, alpha=0.75, zorder=-100)
        plt.show()
        # plot the correspondences
        plt.figure('Registered datasets')
        for index, X in enumerate(PSets[1:]):
            
            plt.subplot(len(PSets)-1,2, 2*index+1)
            plt.title('Unregistered')
            plt.scatter(R[:,0],R[:,1],color='k', s = s, alpha=0.75, zorder=-100)#,clip_on=False)
            plt.scatter(X[:,0],X[:,1],color='r', s = s, alpha=0.75, zorder=-100)
            plt.ylabel('Worm {}'.format(index +1))
            for j, (x,y,z) in enumerate(R):
                if (ids[j,index])>0:
                    plt.plot([x, X[ids[j,index], 0]], [y, X[ids[j,index], 1]], color='gray')
            Xreg = PSets_reg[index]
            plt.subplot(len(PSets)-1,2, 2*index+2)
            plt.title('Registered')
            plt.scatter(R[:,0],R[:,1],color='k', s = s, alpha=0.75, zorder=-100)#,clip_on=False)
            plt.scatter(Xreg[:,0],Xreg[:,1],color='r', s = s, alpha=0.75, zorder=-100)
            for j, (x,y,z) in enumerate(R):
                if (ids[j,index])>0:
                    plt.plot([x, Xreg[ids[j,index], 0]], [y, Xreg[ids[j,index], 1]], color = 'gray')
            
        plt.show()
        # plot the joint heatmaps
        Hmaps = []
        for ind, fname in enumerate(files):
            Ratio = scipy.io.loadmat(fname)['Ratio2']
            if ind == 0:
                order = np.array(scipy.io.loadmat(fname)['cgIdx']).T[0]-1
                try:
                    badNeurs = scipy.io.loadmat(fname)['flagged_neurons'][0]
                    print badNeurs
                    order = np.delete(order, badNeurs)
                except KeyError:
                    pass
            if ind > 0:
                Ratio = Ratio[ids[:,ind-1]]
                print ids[:,ind-1]
            Hmaps.append(Ratio[:,:2000])
        
        # create one big heatmap
        H = np.concatenate(Hmaps, axis = 1)
            
        plt.figure('Joint heatmap')
        plt.imshow(H[order], aspect = 'auto', vmin = -0.5, vmax = 1.0, interpolation = 'none')
        plt.colorbar()
        plt.show()
        
    


if __name__=="__main__":

    main()











