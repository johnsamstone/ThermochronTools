# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:51:26 2017

@author: sjohnstone

Test of MCMC volume calibration for Leah's data

"""

import VolumeCalibration as vc
import loadVCfileFuns as lvc
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import emcee #MCMC sampler
import corner


#==============================================================================
# Define paths to data 
#==============================================================================

path = '/Users/sjohnstone/Documents/HeLab/VolumeCalibration/From_LM/Depletion_9Sept.txt'

#These things don't matter in this case because temperatures are the same - 
# so we are not performing a thermal transpiration correction
gas = 'N2'
pipeDiam = 0.01

#Setup the calibration volume
Vc = 50.00
dVc = 0.02
calVolume = vc.calibrationVolume(Vc,dVc)

#==============================================================================
#Load data 
#==============================================================================

measurements = lvc.LMdataLoader(path,calVolume,pipeDiam,gas)

diluExp = vc.dilutionExperiments(measurements,calVolume)

diluExp.depletionRegression(1.1332,0.07)
diluExp.plotResults()

#%%
#==============================================================================
# Setup the probability distributions for the prior and posterior
#==============================================================================

def log_likelihood(theta):
    return np.log10(diluExp.calcLikelihood(theta[0],theta[1],10**theta[2]))

def log_prior(theta):
    initialPressure,volumeRatio,sigma = theta
    vRatioPrior = 0.99
    dVratioPrior = vRatioPrior*0.2
    
    inPressurePrior = 1.1322
    dInPressure = inPressurePrior * 0.2
    
    
    return (np.log10(stats.norm(vRatioPrior,dVratioPrior).pdf(volumeRatio)) +
            np.log10(stats.norm(inPressurePrior,dInPressure).pdf(initialPressure)))

def log_posterior(theta):
    
    return log_likelihood(theta) + log_prior(theta)

#Setup

#==============================================================================
# Setup parameters for markov chain model
#==============================================================================
#Setup
ndim = 3
nwalkers = 50
nburn = 5e2
nsteps = nburn + 5e2
starting_guesses = np.vstack((1.1322 + np.random.randn(nwalkers)*0.0005,1.0 + np.random.randn(nwalkers)*0.01,np.log10(np.abs(5e-1 + np.random.randn(nwalkers)*1e-2)))).T

sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior)

log_likelihood(starting_guesses[1,:])

#%%
#==============================================================================
# Ry the markov chain model
#==============================================================================
sampler.run_mcmc(starting_guesses,nsteps)

#%%
#==============================================================================
# lot the results
#==============================================================================
samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
for i in range(ndim):
    plt.subplot(ndim,1,i+1)
    plt.hist(samples[:,i],100)
#    for j in range(nwalkers):
#        plt.plot(sampler.chain[j,:,i],'-k')

#%%    
#==============================================================================
#     
#==============================================================================

valueLabels = ('P_0','V ratio','error')

fig = corner.corner(samples,labels = valueLabels)