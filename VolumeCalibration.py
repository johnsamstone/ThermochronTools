# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:00:25 2017

@author: sjohnstone

Functions for performing volume calibration
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


#==============================================================================
# Useful functions
#==============================================================================

def weightedRegression(xs,ys,y_err,slopeGuess,interceptGuess):
    ''' Calculate the best fitting linear fit and return it and the covariance matrix
    '''    
    
    def fitFn(x,m,b):
        return (m*x)+b

    p_bf,p_cov = optimize.curve_fit(fitFn,xs,ys,(slopeGuess,interceptGuess),sigma = y_err,absolute_sigma = True)
    
    return p_bf,p_cov
    
def plotWeightedRegression(xs,ys,y_err,slopeGuess,interceptGuess,**kwargs):
    
    p_bf,p_cov  = weightedRegression(xs,ys,y_err,slopeGuess,interceptGuess)
    
    
    x2plot = np.linspace(np.min(xs),np.max(xs),100)    
    plt.plot(x2plot,p_bf[0]*x2plot + p_bf[1],'-k',linewidth = 2)
    

    plt.plot(xs,ys,'ok')

    
    return p_bf,p_cov
    
def weightedMean(xs,errors):

    weights = 1/(errors**2)

    #Weighting factors
    wFactors = weights/np.sum(weights)

    #weighted mean
    wMean = np.sum(xs*wFactors)
    
    #weighted mean error
    wMeanErr = np.sqrt(np.sum((wFactors*errors)**2))    
    
    return wMean, wMeanErr 


#==============================================================================
# 
#==============================================================================
class calibrationVolume:
    
    def __init__(self,Volume,error):
        
        self.Vc = Volume
        self.dVc = error
        
    def addVolume(self,SecondVolume):
        
        newVc = self.Vc + SecondVolume.Vc
        newError = newVc * np.sqrt((self.dVc/self.Vc)**2 + (SecondVolume.dVc/SecondVolume.Vc)**2)
        
        return calibrationVolume(newVc,newError)


#==============================================================================
# 
#==============================================================================
class pressureMeasurement:
    
    def __init__(self,pressure,measurementError,Tline,Tgauge,thermalTranspirationCorrection):
        
        #Assign variables
        self.uncorrected_pressure = pressure
        self.uncorrected_error = measurementError
        self.Tline = Tline
        self.Tgauge = Tgauge
        self.thermalTranspirationCorrection = thermalTranspirationCorrection        
        
        #Gather data
        self.applyCorrection()
        
        self.calcMeanPressure()        
        
    def applyCorrection(self):
        '''Apply the assigned thermmal transpiration correction
        '''
        self.pressure = self.thermalTranspirationCorrection.applyCorrection(self.uncorrected_pressure,self.Tgauge,self.Tline)
        self.error = self.pressure*(self.uncorrected_error/self.uncorrected_pressure)
        
    def calcMeanPressure(self):
        ''' Assign the mean pressure either based on a weighted mean (if theres more than 1 measurement)
        '''
        if len(self.pressure) > 1:
            self.meanPressure,self.meanPressureError = weightedMean(self.pressure,self.error)
        else:
            self.meanPressure,self.meanPressureError = self.pressure,self.error
            
    def plotWeightedMean(self):
        ''' Visualization of the weighted mean age
        '''
        
        x2Plot = np.arange(len(self.pressure))
        plt.plot(x2Plot,np.ones_like(x2Plot)*(self.meanPressure - self.meanPressureError),'--k')
        plt.plot(x2Plot,np.ones_like(x2Plot)*(self.meanPressure + self.meanPressureError),'--k')
        plt.plot(x2Plot,np.ones_like(x2Plot)*(self.meanPressure),'-k',linewidth = 2)
        plt.errorbar(x2Plot,self.pressure,yerr = self.error,fmt = 'o')       

#==============================================================================
#       
#==============================================================================
class expansionExperiment:
    experimentNumber = None #The number of the experiment
    expansionDirection = None
    Vc = None #The known calibrated volume
    dVc = None #The error on that calibrated volume
   
    P1 = None
    dP1 = None
    P2 = None
    dP2 = None
   
    Vu = None
    dVu = None
    
    def __init__(self,initialPressureMeasurement,expandedPressureMeasurement,
                 thermalTranspirationCorrection,calibrationVolume,expansionDirection = 'Forward'):
        
        #Store initial data
        self.initialMeas = initialPressureMeasurement
        self.expandedMeas = expandedPressureMeasurement
                
        
        #Assign variables
        self.thermalTranspirationCorrection = thermalTranspirationCorrection
        self.expansionDirection = expansionDirection
        
        self.P1,self.dP1 = self.initialMeas.meanPressure, self.initialMeas.meanPressureError
        self.P2,self.dP2 = self.expandedMeas.meanPressure,self.expandedMeas.meanPressureError
        
        self.Vc = calibrationVolume.Vc
        self.dVc = calibrationVolume.dVc
    
        self.calcVolume()
    
    def calcVolume(self):
        
        if self.expansionDirection == 'Forward':
            self._calcVolumeForward()
        elif self.expansionDirection == 'Reverse':
            self._calcVolumeReverse()
        else:
            print('Error, must define expansion direction to be either \'Forward\' or \'Reverse\'')
    
    def _calcVolumeForward(self):
        ''' Calculate the unknown volume based on an experiment where gas at a known
        pressure is expanded from a known volume to an unknown volume. 
        
        e.g.:

        P1Vc = P2(Vc+Vu)

        Vu = (P1Vc/P2) - Vc        
        
        Vu = Vc ((P1/P2) - 1)        
        
        '''             
        
        self.Vu = self.Vc*((self.P1/self.P2) -1.0)

        #### Error propogation                 

        self.dVu = self.Vu * np.sqrt((self.dP1/self.P1)**2 + (self.dP2/self.P2)**2 + (self.dVc/self.Vc)**2)
        
    
    def _calcVolumeReverse(self):
        ''' Calculate the unknown volume based on an experiment where gas at a known
        pressure is expanded from an unknown volume into a known volume. 
        
        e.g.:

        P1Vu = P2(Vc+Vu)

        Vu = Vc/((P1/P2) - 1)        
        
        '''     
        
        self.Vu = self.Vc/((self.P1/self.P2) - 1.0)
    
        ### Error propogation 
        denom = ((self.P1/self.P2) - 1.0)
        dDenom = denom * np.sqrt((self.dP1/self.P1)**2 + (self.dP2/self.P2)**2)   
    
        self.dVu = self.Vu * np.sqrt((self.dVc/self.Vc)**2 + (dDenom/denom)**2)
    
    def calcLikelihood(self,Vunknown,sigma = None):
        ''' returns the likelihood that the guessed unknown volume describes the data
        '''
        
        if sigma is None:
            sigma = self.dVu
            
        return (1/np.sqrt(2.0*sigma**2*np.pi))*np.exp(-(Vunknown - self.Vu)**2/(2*sigma**2))

#==============================================================================
# 
#==============================================================================
class expansionExperimentSet:
    
    _experiments = None
    _nExperiments = 0
    
    def __init__(self,experiments):
        self._experiments = experiments
        self._nExperiments = len(experiments)

    def flattenParameter(self,parameter):
        ''''Returns a python list of the concatenated values of the value of
        'parameter' in each of the experiments stored within the set. 
        '''
        paramOut = []        
        for exp in self._experiments:
            paramOut.append(getattr(exp,parameter))
        
        return paramOut
    
    def calcProbabilityDistribution(self,Vaxis = None):
        ''' Calculate the probability distribution of volumes
        '''
        pass
    
    
    def plotProbDistrubution(self):
        '''Plot the pdf
        '''
        
        Vus = np.array(self.flattenParameter('Vu'))
        dVus = 2.0*np.array(self.flattenParameter('dVu'))
        
        VolAxis = np.linspace(np.min(Vus)*0.75,np.max(Vus)*1.25,200)
        p = np.zeros_like(VolAxis)        
        for i in range(len(Vus)):
            p+= (1/np.sqrt(2.0*dVus[i]**2*np.pi))*np.exp(-(VolAxis - Vus[i])**2/(2*dVus[i]**2))
            
        plt.plot(VolAxis,p,'-')
        plt.xlabel('Volume',fontsize = 14)
    
    def calcWeightedMean(self):
        ''' calculate the weighted mean age
        '''
        
        Vus = np.array(self.flattenParameter('Vu'))
        dVus = np.array(self.flattenParameter('dVu'))
        
        self.mean, self.error  = weightedMean(Vus,dVus)
        
        pass
    
    def plotWeightedMean(self):
        ''' Visualization of the weighted mean age
        '''
        self.calcWeightedMean()
        
        Vus = np.array(self.flattenParameter('Vu'))
        dVus = np.array(self.flattenParameter('dVu'))
        
        x2Plot = np.arange(len(Vus))
        plt.plot(x2Plot,np.ones_like(x2Plot)*(self.mean - self.error),'--k')
        plt.plot(x2Plot,np.ones_like(x2Plot)*(self.mean + self.error),'--k')
        plt.plot(x2Plot,np.ones_like(x2Plot)*(self.mean),'-k',linewidth = 2)
        plt.errorbar(x2Plot,Vus,yerr = dVus,fmt = 'o')       

    
    def calcLikelihood(self,Vunknown,sigma = None):
        ''' Calculate the combined likelihood of the unknown volume for all the
        experiements within this set
        '''
        
        L = 1.0
        
        for exp in self._experiments:
            L*=exp.calcLikelihood(Vunknown,sigma)
        
        return L
            
    
class dilutionExperiments:
    
    def __init__(self,measurements,calibrationVolume):
    
        self.calibrationVolume = calibrationVolume
        self._measurements = measurements
        self.ns = np.arange(len(measurements))*1.0
        self.Ps = np.array(self.flattenParameter('meanPressure'))
        self.dPs = np.array(self.flattenParameter('meanPressureError'))

    def calculateVolume(self):
        pass
    
    def plotResults(self):
        '''Plot the best fitting results'''
        
        if self.Vratio is None:
            print('Whoops, must performt eh regression before plotting it')
        else:
            plt.errorbar(self.ns,self.Ps,yerr=self.dPs,fmt = 'o')
            plt.plot(self.ns,self.initialPressure*self.Vratio**self.ns,'-k',linewidth = 2)
            plt.yscale('log')

            plt.xlabel('Expansion number')
            plt.ylabel('Pressure')
    
    def flattenParameter(self,parameter):
        ''''Returns a python list of the concatenated values of the value of
        'parameter' in each of the experiments stored within the set. 
        '''
        paramOut = []        
        for meas in self._measurements:
            paramOut.append(getattr(meas,parameter))
        
        return paramOut
    
    def calcLikelihood(self,InitialPressure,Vratio,sigma = None):
        ''' theta = (InitialPressure,Volume ratio)
        '''

        if sigma is None:
            sigma = self.dPs
    
        PressModel = InitialPressure*(Vratio)**self.ns
        
        #Likelihood of the model
        return np.prod(np.exp(-(self.Ps-PressModel)**2/(2.0*sigma**2))/np.sqrt(2.0*np.pi*sigma**2))

    
    def depletionRegression(self,InitialPressureGuess,unknownVolumeGuess):
    
        
        func = self.expectedDepletion #Function being used in curve fit

        initialGuess = (self.calibrationVolume.Vc/(unknownVolumeGuess + self.calibrationVolume.Vc), InitialPressureGuess)
        
        betaHat, self.cov = optimize.curve_fit(func, self.ns, self.Ps, initialGuess, sigma=self.dPs, absolute_sigma=True)
        
        self.Vratio = betaHat[0]
        self.initialPressure = betaHat[1]
        self.dVratio = self.cov[0,0]**0.5
        self.dInitialPressure = self.cov[1,1]**0.5
        
        Vboth = self.calibrationVolume.Vc/self.Vratio
        dVboth = Vboth * np.sqrt((self.calibrationVolume.dVc/self.calibrationVolume.Vc)**2 + (self.dVratio/self.Vratio)**2)
        
        self.Vu = Vboth - self.calibrationVolume.Vc
        self.dVu = np.sqrt(dVboth**2 + self.calibrationVolume.dVc**2)
    
    
    def logExpectedDepletion(self,n,logVratio,logInitialPressure):
        return logInitialPressure + n*logVratio
    
    def expectedDepletion(self,n,Vratio,initialPressure):
        return initialPressure*Vratio**n