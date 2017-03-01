# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:59:11 2017

@author: sjohnstone
"""

import numpy as np
from matplotlib import pyplot as plt
import VolumeCalibration as VC
import thermalTranspirationCorrection as ttc
import loadVCfileFuns
import emcee
from scipy import stats

#==============================================================================
# Define paths, other info 
#==============================================================================

fillAppPath = ('/Users/sjohnstone/Documents/HeLab/VolumeCalibration/RawData/VolCal_Pip3_Vapp_UnkToCal.txt',
               '/Users/sjohnstone/Documents/HeLab/VolumeCalibration/RawData/VolCal_Pip3_Vapp_UnkToCal_2.txt')

pipPath = ('/Users/sjohnstone/Documents/HeLab/VolumeCalibration/RawData/VolCal_Pip3_pip_CalToUnk_2.txt',
           '/Users/sjohnstone/Documents/HeLab/VolumeCalibration/RawData/VolCal_Pip3_pip_CalToUnk_2.txt',
           '/Users/sjohnstone/Documents/HeLab/VolumeCalibration/RawData/VolCal_Pip3_pip_CalToUnk_3.txt')

           


nIters = 500 #How many monte carlo steps to take in the MCMC

Vc = 51.639
dVc = 0.068

diam = 0.01

calVolume = VC.calibrationVolume(Vc,dVc)

tempGetter = loadVCfileFuns.tempGetter()
#%%
#==============================================================================
# Load the data 
#==============================================================================

expSet_fillingApparatus = loadVCfileFuns.SJdataLoader(fillAppPath,calVolume,tempGetter,diam,'N2','Reverse')

expSet_fillingApparatus.calcWeightedMean()
plt.figure()
expSet_fillingApparatus.plotWeightedMean()
plt.title('Filling Apparatus: Pippette 3')
plt.ylabel(r'Volume cm$^3$')
plt.xlabel('Experiment number')


#%%
#==============================================================================
# Setup Bayesian model
#==============================================================================

Vprior = expSet_fillingApparatus.mean
dVprior = Vprior*0.20

ErrorGuess = 5e-1
ErrorGuessRange = ErrorGuess*0.2

def log_likelihood(theta):
    V,sigma = theta
    return np.log10(expSet_fillingApparatus.calcLikelihood(V,10**sigma))

def log_prior(theta):
    V,sigma = theta    
    
    return np.log10(stats.norm(Vprior,dVprior).pdf(V))

def log_posterior(theta):
    
    return log_likelihood(theta) + log_prior(theta)

#Setup
ndim = 2
nwalkers = 50
nburn = 5e2
nsteps = nburn + nIters

VolumeStartingGuess = np.random.normal(Vprior,dVprior/50,size = (nwalkers,1))

#Rather than drawing random error look at the log of errors (this keeps values positive)
PressureErrorStartingGuess = np.log10(np.abs(np.random.normal(ErrorGuess,ErrorGuessRange,size = (nwalkers,1))))

starting_guesses = np.hstack((VolumeStartingGuess,PressureErrorStartingGuess))

#%%
#==============================================================================
# Run bayesian 
#==============================================================================
sampler_app = emcee.EnsembleSampler(nwalkers,ndim,log_posterior)

sampler_app.run_mcmc(starting_guesses,nsteps)

# chain is of shape (nwalkers, nsteps, ndim): # discard burn-in points and reshape:
trace = sampler_app.chain[:, nburn:, :]
trace = trace.reshape(-1,ndim).T

#%%
#==============================================================================
# Plot the results
#==============================================================================

plt.figure()
ValueNames = (r'Volume [cm$^3$]', 'log(Measurement error [mTorr])')
samples_app = sampler_app.chain[:, nburn:, :].reshape((-1, ndim))
for i in range(ndim):
    plt.subplot(ndim,1,i+1)
    plt.hist(samples_app[:,i],100)
    plt.xlabel(ValueNames[i],fontsize = 14)
#    for j in range(nwalkers):
#        plt.plot(sampler_app.chain[j,:,i],'-')
#        plt.ylabel(ValueNames[i])
    
    p50 = np.percentile(samples_app[:,i],50)
    p13 = np.percentile(samples_app[:,i],13.6)
    p86 = np.percentile(samples_app[:,i],86.4)

    print(ValueNames[i]+' %.3f, + %.3f, - %.3f'%(p50,p50-p13,p86-p50))
 
plt.subplot(ndim,1,1)
plt.title('Filling Apparatus: Pippette 3')
   


#%%
#==============================================================================
# Propogate this through to the pipette
#==============================================================================

fillingApparatusVolume = VC.calibrationVolume(np.percentile(samples_app[:,1],50),np.std(samples_app[:,1]))

newVolume = calVolume.addVolume(fillingApparatusVolume)

expSet_pippette = loadVCfileFuns.SJdataLoader(pipPath,newVolume,tempGetter,diam,'N2','Forward')

plt.figure()
expSet_pippette.plotWeightedMean()
plt.title('Filling Apparatus: Pippette 3')
plt.ylabel(r'Volume cm$^3$')
plt.xlabel('Experiment number')
#%%
#==============================================================================
# Setup Bayesian model
#==============================================================================

Vprior = 0.03
dVprior = Vprior*0.2

ErrorGuess = np.percentile(samples_app[:,1],50)
ErrorGuessRange = np.std(samples_app[:,1])

def log_likelihood(theta):
    V,sigma = theta
    return np.log10(expSet_pippette.calcLikelihood(V,10**sigma))

def log_prior(theta):
    V,sigma = theta    
    
    return np.log10(stats.norm(Vprior,dVprior).pdf(V))

def log_posterior(theta):
    return log_likelihood(theta) + log_prior(theta)

#Setup
ndim = 2
nwalkers = 50
nburn = 5e2
nsteps = nburn + nIters

VolumeStartingGuess = np.random.normal(Vprior,dVprior/50,size = (nwalkers,1))

#Rather than drawing random error look at the log of errors (this keeps values positive)
PressureErrorStartingGuess = np.random.normal(ErrorGuess,ErrorGuessRange,size = (nwalkers,1))

starting_guesses = np.hstack((VolumeStartingGuess,PressureErrorStartingGuess))

#%%
#==============================================================================
# Run bayesian 
#==============================================================================
sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior)

sampler.run_mcmc(starting_guesses,nsteps)

# chain is of shape (nwalkers, nsteps, ndim): # discard burn-in points and reshape:
trace = sampler.chain[:, nburn:, :]
trace = trace.reshape(-1,ndim).T

#%%
#==============================================================================
# Plot the results
#==============================================================================

plt.figure()
ValueNames = (r'Volume [cm$^3$]', 'log(Measurement error [mTorr])')
samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
for i in range(ndim):
    plt.subplot(ndim,1,i+1)
    plt.hist(samples[:,i],100)
    plt.xlabel(ValueNames[i],fontsize = 14)
#    for j in range(nwalkers):
#        plt.plot(sampler.chain[j,:,i],'-')
#        plt.ylabel(ValueNames[i],fontsize = 14)
    p50 = np.percentile(samples[:,i],50)
    p13 = np.percentile(samples[:,i],13.6)
    p86 = np.percentile(samples[:,i],86.4)

    print(ValueNames[i]+' %.3f, + %.3f, - %.3f'%(p50,p50-p13,p86-p50))
    
plt.subplot(ndim,1,1)
plt.title('Pippette 3')