# -*- coding: utf-8 -*-
'''
Created on Thu Dec 15 15:59:53 2016

@author: sjohnstone
 
 This is a class that describes a thermochronometer

'''

__author__ = ["Sam Johnstone"]
__copyright__ = "2016"
__credits__ = ["Sam Johnstone", "Rob Sare"]
__license__ = "MIT"
__maintainer__ = "Sam Johnstone"
__email__ = "sjohnstone@usgs.gov"
__status__ = "Production"



import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

#==============================================================================
# Constants used throughout
#==============================================================================
R = # Universal gas constant
Av = #Avogadros number


#==============================================================================
#  Classes for different thermochronometers
#==============================================================================

class Thermochronometer():
    
    '''
    '''
    _parents = None
    _daughters = None
    _decayConsts = None
    _daughterProductionFactors = None
    
    __multipleParents = None #is there more than one parent
        
    def __init__():
        pass
    
    def calcAge(self):
        
        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.
    
        If there are multiple parents, this is done via iterative, root finding methods. 
        If there is a single parent the age is calculated directly.      
        
        
        NOTE: Should add some checks to make sure that we don't try and calculate anything without
        first defining values
        '''
        if self.__multipleParents:
            
            return self._calcAgeMultipleParents()
        
        else:
        
            return self._calcAgeSingleParent()
            
           
    def _calcAgeMultipleParents(self,t0 = 1e6):
               
        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.
    
        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods      
        
        '''           
           
        def rootFunc(self,t):
                
            sumParentsRemaining = np.zeros_like(self._daughter)
            sumDecayComponent = np.zeros_like(self._daughter)    
            for i,f in enumerate(self._daughterProductionFactors):
                sumParentsRemaining+=f*self._parents[i]
                sumDecayComponent+=f*self._parents[i]*np.exp(self._decayConsts[i]*t)
            
            return (self._daughters + sumParentsRemaining) - sumDecayComponent
        
        tErr = lambda t: rootFunc(self._parents,self._daughters,self._decayConsts,self._productionFactors,t)
        t0 = 1e6*np.ones_like(self._daughter) #Initial guess of 1 MA   
            
        return optimize.root(tErr,t0).x
        
    def _calcAgeSingleParent(self):
           
        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.
    
        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods      
        
        ''' 
    
        return (1/self._decayConsts)*np.log(1+self._daughters/(self.daughterProductionFactors*self._parents))          
        
    def calcDecayProductionRate(self):
        '''Rate of atoms of a parent lost for
        the decay constant lam and the number of parent atoms nParents
        
        dN/dt = -lam*N
        dD/dt = f*lam*N (where f is the number of daughters produced per parent decay)
        
        RETURNS:
        
        dN/dt - parent loss rate (negative)
        dD/dt -daughter accumulation rate (positive)
        '''
        
        if self.__multipleParents:
            dNdt,dDdt = self._multiDecayProductionRate(self)
        else:
            dNdt = -self.decayConts*self._parents
            dDdt = self._productionFactors*dNdt
        
        return dNdt,dDdt

    def _multiDecayRate(self):
        ''' For when multiple parents produce the same daughter, this calculates
        the rates of parent loss and daughter gain
        
        NOTE: SHOULD THINK ABOUT HOW I AM PASSING AND STORING THE PARENT CONC
        AND THE PRODUCTION RATE OF THAT CONC        
        
        '''    
        dDdt = np.zeros_like(parentConcs[0])#Preallocate space for the daughter production rate   
        dNdt = [] #initialize a list for the parent decay rates
        
        for i,f in enumerate(productionFactors):
            dN_idt = -decayConsts[i]*parentConcs[i]
            
            dD_idt = productionFactors[i]*dN_idt

            dDdt+= dD_idt
            dNdt.append(dN_idt)
        
        
        return np.array(dNdt),dDdt        
        
class HeThermochronometer(Thermochronometer):
    ''' A thermochronometer where the daughter product is He produced by decay of U, Th, Sm
    and daughter is lost by thermally activated diffusion
    '''    
    
    def __init__():
        pass
    
    def alphaEjection(self): 
        ''' NOTE: Hmmmm.... how exactly do I want to do this... perhaps an alpha ejection function should be its own class?
            Or maybe a crystal model is its own class? It can have different types.... e.g. spherical, cyclindrical, apatite, zircon
            but then perhaps these would also need their own concentrations of parent and daughter - to deal with zonation?
            
            or maybe I can build in flexibility to allow the grain model to overwrite the concentrations of the class?
            
            perhaps I want to have a a redifinition of the calc production rate function that 
            implements some ejection?
        '''
        pass
    
    def getDiffusivity(self,T):
        
        ''' given the diffusivity specific to an individual mineral system 
        (defined by sub class), what is the diffusivty at the perscribed temperature
        
        NOTE!!! THERE IS PROBABLY A CLEANER WAY TO DO THIS        
        
        '''
        
        return self._Do*np.exp(-self._Ea/(R*T))

class ApatiteHe(HeThermochronometer):
    '''A class describing the apatite U-Th/He thermochronometer
    '''

    #Default diffusivities from Shuster et al., ...
    self._Do = 1#
    self._Ea = 1#
    
    def __init__(U,Th,Sm,He = 0.0,grainModel):
        
    