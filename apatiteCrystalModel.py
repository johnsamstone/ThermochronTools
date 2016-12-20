import numpy as np
from scipy import optimize

#==============================================================================
# Thermal diffusivity
#==============================================================================

def thermDiffusivity(T,Do,Ea,R):
    '''Temperature dependent diffusivity, D = Do*exp(-Ea/RT) 
    
        Remember to be careful with units, T in Kelvin, R in J K^-1 mol^-1, 
        Ea in J
    '''    
    
    return Do*np.exp(Ea/(R*T))


def diffusion(N,dx,T,diffusivityFunction):
    ''' Calculates the flux of daughter product following the temperature dependent
    diffusivity specified by the temperature T and the function of temperature
    diffusivityFunction. E.g. 
    
    dNdT = D_0*exp(-Ea/RT) * grad^2 N 
    
    NOTES!!!: Here dNdT will be a smaller array then N, this function
    does not handle boundary conditions. These should be handled either be creating
    Pseudo-nodes for the N array or explicitly treating boundaries at a later step    
    
    '''
    
    if N.ndim == 1:
        grad2N = (N[2:] - 2.0*N[1:-1] + N[:-2])/(dx**2)
    elif N.ndim ==2:
        grad2N_x = (N[1:-1,2:] - 2.0*N[1:-1,1:-1] + N[1:-1,:-2])/(dx**2)
        grad2N_y = (N[2:,1:-1] - 2.0*N[1:-1,1:-1] + N[:-2,1:-1])/(dx**2)
        grad2N = grad2N_x + grad2N_y
        
    return  diffusivityFunction(T) * grad2N
    
#==============================================================================
#Production equations 
#==============================================================================

def singleProductionFunction(nParents,lam,t):
    '''Number of atoms of a daughter produced over the time interval t, for
    the decay constant lam and the number f parent atoms nParents
    '''
    return nParents * (np.exp(-lam*t))

def multiProductionFunction(parentConcs,decayConsts,productionFactors,t):
    ''' For when multiple parents produce the same daughter, this calculates
    total production over the time interval t
    
        Inputs:
        
        parentConcs = number of parent atoms, a tuple or tuple of arrays
        decayConsts = the decay constants for those parents, a tuple
        productionFactors = the number of daughters each parent decay chain produces, a tuple
    '''    
    daughter = np.zeros_like(parentConcs[0])#Preallocate space for the daughter    
    
    for i,f in enumerate(productionFactors):
        newParents = singleProductionFunction(parentConcs[i],decayConsts[i],t)
        daughter+= f*(newParents - parentConcs[i])
        parentConcs[i] = newParents
    
    
    return parentConcs,daughter
    
def singleDecayRate(nParents,lam):
    '''Rate of atoms of a parent lost for
    the decay constant lam and the number of parent atoms nParents
    
    dN/dt = -lam*N    
    
    '''
    return -lam * nParents

def multiDecayRate(parentConcs,decayConsts,productionFactors,redistributionFunction = None):
    ''' For when multiple parents produce the same daughter, this calculates
    the rates of parent loss and daughter gain
    
        Inputs:
        
        parentConcs = number of parent atoms, a tuple or tuple of arrays
        decayConsts = the decay constants for those parents, a tuple
        productionFactors = the number of daughters each parent decay chain produces, a tuple
    '''    
    dDdt = np.zeros_like(parentConcs[0])#Preallocate space for the daughter production rate   
    dNdt = [] #initialize a list for the parent decay rates
    
    for i,f in enumerate(productionFactors):
        dN_idt = singleDecayRate(parentConcs[i],decayConsts[i])
        
        dD_idt = -productionFactors[i]*dN_idt
        
        #If this decay scheme requires redistribution (e.g. alpha ejection)
        if not(redistributionFunction is None):
            dD_idt = redistributionFunction(dD_idt,i)

        dDdt+= dD_idt
        dNdt.append(dN_idt)
    
    
    return dNdt,dDdt
    
    
#==============================================================================
# Redistribute daughter
#==============================================================================

def alphaRedistribution(production,coordinates,grainRadius,relocationdistance):
    '''NOTE!!! THIS TOTALLY NEEDS TO BE REVISED, JUST TRYING A SIMPLE FORMULATION FOR 
    SPHERICAL COORDINATES (e.g. Farley), in order to perform a thought experiment.
    '''    
    
    redistribution = np.zeros_like(production)    
    for i,Xo in enumerate(coordinates):
        if (grainRadius - relocationdistance) < Xo:
            Xstr = (1.0/(2.0*Xo))*((Xo**2) + (grainRadius**2) - (relocationdistance**2))
            redistribution[i] = 0.5 + ((Xstr - Xo)/(2*relocationdistance))
        else:
            redistribution[i] = 1.0
            
    return production*redistribution

#==============================================================================
# Age calculation
#==============================================================================

def calcAge(parents,daughter,decayConsts,productionFactors):
    '''DEFINITELY NEED TO WRITE DESCRIPTION
    '''
    
    def rootFunc(parents,daughter,decayConsts,productionFactors,t):
        
        sumParentsRemaining = np.zeros_like(daughter)
        sumDecayComponent = np.zeros_like(daughter)    
        for i,f in enumerate(productionFactors):
            sumParentsRemaining+=f*parents[i]
            sumDecayComponent+=f*parents[i]*np.exp(decayConsts[i]*t)
        
        return (daughter + sumParentsRemaining) - sumDecayComponent
    
    tErr = lambda t: rootFunc(parents,daughter,decayConsts,productionFactors,t)
    t0 = 1e6*np.ones_like(daughter)   
    
    return optimize.root(tErr,t0).x
    
    
#==============================================================================
# Time temperature variations
#==============================================================================


def tempAtTime(t):
    '''Constant temp for test'''
    return 273.15 + 10.0
