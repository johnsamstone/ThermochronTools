# -*- coding: utf-8 -*-
'''
Created on Thu Dec 15 15:59:53 2016

@author: sjohnstone
 
 This is a class that describes a thermochronometer

'''

__author__ = ["Sam Johnstone"]
__copyright__ = "2016"
__credits__ = ["Sam Johnstone"]
__license__ = "MIT"
__maintainer__ = "Sam Johnstone"
__email__ = "sjohnstone@usgs.gov"
__status__ = "Production"



import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.sparse import diags
from scipy import linalg

#==============================================================================
# Constants used throughout
#==============================================================================
R = 8.0 # Universal gas constant
Av = 6.022#Avogadros number


#==============================================================================
# Useful functions
#==============================================================================
def thermDiffusivity(T,Do,Ea,R):
    '''Temperature dependent diffusivity, D = Do*exp(-Ea/RT) 
    
        Remember to be careful with units, T in Kelvin, R in J K^-1 mol^-1, 
        Ea in J
    '''    
    
    return Do*np.exp(Ea/(R*T))
    
#==============================================================================
# Class for a grain model
#==============================================================================

class grainModel():
    ''' A grain model handles the calculation of parameters that may be specific to thermochronometer shape.
    This includes the diffusive loss of daughter product, redistribution and/or ejection
    of daughter products, etc. These will be handled by sub classes
    '''
    
    def __init__():
        pass
    
    def calcDaughterLossRate(self,D):
        ''' Calculate the rate of daughter product loss, e.g. due to diffusion
        D (m^2/s) specifies the diffusivity
        '''
        
        print('No specific grain geometry specified, cannot calculate daughter loss')
        return None
        
    def redistributeProductionRate(self,daughterProductionRate):
        ''' Redistributes the produced daughter product according to the relocation
        distance of different nuclides specied by the subclasses
        '''
        
        print('No specific grain geometry specified, cannot calculate daughter redistribution')
        
def sphere(grainModel):
    '''
    '''
    def __init__():
        pass        
    
def prism(grainModel):
    
    def __init__():
        pass

def apatite(grainModel):
    
    def __init__():
        pass
    
def zircon(grainModel):
    
    def __init__():
        pass

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
    _diffusivityFunction = None # Afunction that excepts a teperature and returns a diffusivity
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
        
        tErr = lambda t: rootFunc(self,t)
        t0 = 1e6*np.ones_like(self._daughters) #Initial guess of 1 MA   
            
        return optimize.root(tErr,t0).x
        
    def _calcIntegratedAgeMultipleParents(self,t0 = 1e6):
               
        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.
    
        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods      
        
        '''           
           
        def rootFunc(self,t):
                
            sumParentsRemaining = np.zeros_like(self._daughters)
            sumDecayComponent = np.zeros_like(self._daughters)    
            for i,f in enumerate(self._daughterProductionFactors):
                sumParentsRemaining+=f*np.sum(self._parents[i])
                sumDecayComponent+=f*np.sum(self._parents[i]*np.exp(self._decayConsts[i]*t))
            
            return (self._daughters + sumParentsRemaining) - sumDecayComponent
        
        tErr = lambda t: rootFunc(self,t)
        t0 = 1e6*np.ones_like(self._daughters) #Initial guess of 1 MA   
            
        return optimize.root(tErr,t0).x
        
    def _calcAgeSingleParent(self):
           
        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.
    
        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods      
        
        ''' 
    
        return (1/self._decayConsts)*np.log(1+self._daughters/(self._daughterProductionFactors*self._parents))          
       
    def _calcIntegratedAgeSingleParent(self):
           
        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.
    
        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods      
        
        ''' 
    
        return (1/self._decayConsts)*np.log(1+np.sum(self._daughters)/(self._daughterProductionFactors*np.sum(self._parents)))          
       
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
            dNdt = -self._decayConsts*self._parents
            dDdt = -self._daughterProductionFactors*dNdt
        
        return dNdt,dDdt
        
    def calcDaughterLossRate(self,T):
        
        ''' Calculates the rate of loss of daughter product at the specified temperature.
        The actual function for this will be defined by subclasses. All thermochronometers
        have a temperature dependent loss of daughter product, but they do not all 
        share the same depdenencies. 
        '''
        print('Error: calcDaughterLossRate not defined by subclass')
        return None

    def _multiDecayRate(self):
        ''' For when multiple parents produce the same daughter, this calculates
        the rates of parent loss and daughter gain
        
        NOTE: SHOULD THINK ABOUT HOW I AM PASSING AND STORING THE PARENT CONC
        AND THE PRODUCTION RATE OF THAT CONC        
        
        '''    
        dDdt = np.zeros_like(self._parents[0])#Preallocate space for the daughter production rate   
        dNdt = [] #initialize a list for the parent decay rates
        
        for i,f in enumerate(self._daughterProductionFactors):
            dN_idt = -self._decayConsts[i]*self._parents[i]
            
            dD_idt = -f*dN_idt

            dDdt+= dD_idt
            dNdt.append(dN_idt)
        
        
        return np.array(dNdt),dDdt        


class sphericalThermochronometer(Thermochronometer):
    ''' A Basic spherical thermochronometer
    '''
    def __init__(self, radius, dr, diffusivityFunction, parentConcs, daughterConcs, decayConstants,daughterProductionFactors):
        '''

        :param radius:
        :param dr:
        :param diffusivityFunction:
        :param parentConcs:
        :param daughterConcs:
        :param decayConstants:
        :param daughterProductionFactors:
        '''

        self.radius
        self.dr = dr
        self.rs = np.arange(dr / 2.0, radius + dr, dr)
        self._n = len(self.rs)
        self._diffusivityFunction = diffusivityFunction

        self._daughters = daughterConcs
        self._parents = parentConcs

        self._decayConsts = decayConstants
        self._daughterProductionFactors = daughterProductionFactors
        self.external_bc = 0.0

        # Set up matrices for solution
        a = np.array([np.ones(self._n - 1), np.ones(self._n), np.ones(self._n - 1)])

        # Create placeholders for matrices used for implicit solution (following Ketcham 2005)
        self._M = diags(a, [-1, 0, 1])
        self._N = diags(-a, [-1, 0, 1])
        self._prodMatrix = diags(a,[-1,0,1])
        self._diagIndices = np.diag_indices(self._n)

        self._ejectionFraction = self._calcEjectionFraction()

    def _calcEjectionFraction(self):
        '''
        No ejection for this thermochronometer
        :return:
        '''

        np.ones_like(self.rs)

    def calcDaughterLossRate(self,T):
        ''' Calculates the flux of daughter product following the temperature dependent
        diffusivity specified by the temperature T and the function of temperature
        diffusivityFunction. E.g. 
        
        dNdT = D_0*exp(-Ea/RT) * grad^2 N 
        '''
        
        #Apply boundary conditions - reflective and 0 concentration (outside grain)
        N = np.hstack((self._daughters[1],self._daughters,self.external_bc))

        grad2N = (N[2:] - 2.0*N[1:-1] + N[:-2])/(self.dr**2)

        return  self._diffusivityFunction(T) * grad2N
        
    
    def integrateTimestep(self,T,dt):
        ''' Integrates the production diffusion equation
        '''
        dNdt,dDdt = self.calcDecayProductionRate()
        dDdt*=self._ejectionFraction
        self._parents+=dNdt*dt #Could make this more sophisticated

        #Set up linear algebra solution
        b = 2.0*self.dr**2/(self._diffusivityFunction(T)*dt)
        self._M[self._diagIndices] = -(b+2.0)
        self._N[self._diagIndices] = (2.0 - b)
        A = -dDdt*self.r*b*dt
        self._prodMatrix = diags(np.array([A[:-1],A,A[1:]]))

        sum_RHS = self._prodMatrix + (linalg.dot(self._N,self._daughters*self.rs))


        #Set boundary condition for external node
        sum_RHS[-1,:] = 0
        sum_RHS[-1,-1] = self._daughters[-1]
        self._M[-1,:] = 0
        self._M[-1,-1] = 1

        #Set boundary condition for central node
        sum_RHS[0,0] = -A[0] - (3.0 - b)*self._daughters[0]*self.rs[0]
        sum_RHS[0,1] = -A[0] - self._daughters[1]*self._daughters[0]*self.rs[1]

        self._daughters=np.dot(sum_RHS,linalg.inv(self._M))/self.rs


class SphericalHeThermochronometer(Thermochronometer):
    ''' A thermochronometer where the daughter product is He produced by decay of U, Th, Sm
    and daughter is lost by thermally activated diffusion
    '''

    stoppingDistance = 20.0/1e6 #Stopping distance in meters

    def _calcEjectionFraction(self):
        '''
        Eject He based on the stopping distance and grain size, utilizes the formulation of Farley for a spherical grain
        as summarized by Ketcham (2005)
        :return:
        '''

        Xstr = (self.rs**2 + self.radius**2 - self.stoppingDistance**2)/(2.0*self.rs)
        return 0.5 + (Xstr - self.rs)/(2.0*self.stoppingDistance)
