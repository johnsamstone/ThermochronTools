# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:39:54 2017

@author: sjohnstone
"""

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
import thermalTranspirationCorrection as ttc


0.993887

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

allExperiments = []

thermTransCorr = ttc.transpirationCorrection(gas,pipeDiam)

for i in range(len(measurements)-1):
    
    
    initialMeasurement = measurements[i]
    expandedMeasurement = measurements[i+1]
    

    #Use these to create an expansion experiment
    thisExperiment = vc.expansionExperiment(initialMeasurement,expandedMeasurement,thermTransCorr,calVolume,'Forward')
    allExperiments.append(thisExperiment)    

expSet = vc.expansionExperimentSet(allExperiments)

expSet.plotWeightedMean()

#%%
#==============================================================================
# Setup the probability distributions for the prior and posterior
#==============================================================================

def log_likelihood(theta):
    Vc,sigma = theta
    return np.log10(expSet.calcLikelihood(Vc,10**sigma))

def log_prior(theta):
    Vc,sigma = theta
    V_prior = 0.3
    dV_prior = V_prior*0.1
    
    return np.log10(stats.norm(V_prior,dV_prior).pdf(Vc))

def log_posterior(theta):
    
    return log_likelihood(theta) + log_prior(theta)

#Setup

#==============================================================================
# Setup parameters for markov chain model
#==============================================================================
#Setup
ndim = 2
nwalkers = 50
nburn = 5e2
nsteps = nburn + 5e2
starting_guesses = np.vstack((0.3 + np.random.randn(nwalkers)*0.001,np.log10(np.abs(1e-2 + np.random.randn(nwalkers)*5e-3)))).T
sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior)

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