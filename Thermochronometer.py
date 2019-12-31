# -*- coding: utf-8 -*-
'''
Created on Thu Dec 15 15:59:53 2016

@author: sjohnstone, sjohnstone@usgs.gov

 This is a class that describes a thermochronometer. It is currently only configured for spherical apatite He thermochronometers.

'''

__author__ = ["Sam Johnstone"]
__copyright__ = "2016"
__credits__ = ["Sam Johnstone"]
__license__ = "MIT"
__maintainer__ = "Sam Johnstone"
__email__ = "sjohnstone@usgs.gov"
__status__ = "Production"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.sparse import diags
from scipy import linalg
from scipy import interpolate
from matplotlib import cm
from scipy import integrate
from scipy import stats
from scipy import special


# ==============================================================================
# Constants used throughout
# ==============================================================================
R_CONST = 8.3144598  # Universal gas constant
AV_CONST = 6.022  # Avogadros number


# ==============================================================================
# Useful functions
# ==============================================================================
def thermDiffusivity(T, Do, Ea, R):
    '''Temperature dependent diffusivity, D = Do*exp(-Ea/RT)

        Remember to be careful with units, T in Kelvin, R in J K^-1 mol^-1,
        Ea in J
    '''

    return Do * np.exp(Ea / (R * T))


def kth_diag_indices(a, k):
    '''
    Get the offset diagonal indices at the kth position
    :param a: the matrix
    :param k: relative position of diagonal (k = 0 is diag, k = -1 is below diag, k = 1 is above diag)
    :return:
    '''
    rowidx, colidx = np.diag_indices_from(a)
    colidx = colidx.copy()  # rowidx and colidx share the same buffer

    if k == 0:
        print('Warning! Nothing will be returned, use np.diag_indices for k  = 0')

    if k > 0:
        colidx += k
    else:
        rowidx -= k
    k = np.abs(k)

    return rowidx[:-k], colidx[:-k]


def calcFt(radius, stoppingDistance):
    '''
    Calculate the alpha ejection correction of Farley, 1996
    :return: Ft, the fraction of alpha particles that would be retained
    '''

    return 1 - (3.0 * stoppingDistance) / (4.0 * radius) + stoppingDistance ** 3 / (
    16.0 * radius ** 3)

def approxRadiusFromFt(Ft,stoppingDistance, radiusGuess):
    '''

    :param stoppingDitance:
    :param radiusGuess:
    :return:
    '''

    rootfun = lambda r : Ft - calcFt(r,stoppingDistance)

    return optimize.newton(rootfun,radiusGuess)

def sphericalVolumeIntegral(quantity,radialPositions):
    integralQuantity = 4.0 * np.pi * quantity * radialPositions ** 2
    return integrate.trapz(integralQuantity,radialPositions)


def CJDiffuse(C, int_tao, r, a):
    '''
    Fourier transform based solution to diffusion in a sphere from carlslaw and Jaeger
    NOTE: Use a regular grid of points, and do not cumulatively integrate to avoid instabilities.
    That is - integrate tao for the history you wish to simulate, rather than stepping through a series
    of tao values. For example, in degassing experiments don't treat each 'step' individually, but
    solve for remaining gas considering initial conditions and ALL previous step heating experiments
    Also NOTE: Solution divides by r, so a point at the center of crystal (r=0) will cause problems
    :param C:
    :param int_tao: The integral of the diffusivity over time
    :param r:
    :param a:
    :return:
    '''

    #Do I need to sum over odd interval?
    n = len(C)
    # if np.mod(n,2)==0:
    #     n+=1
    Ks = np.arange(1, n)

    expTerm = lambda k: np.exp(-(k ** 2) * (np.pi ** 2) * int_tao)
    sinTerm = lambda k: np.sin(k * np.pi * r / a)
    integralTerm = lambda k: integrate.trapz(sinTerm(k) * r * C, r)

    C_diffed = np.zeros_like(C)

    for k in Ks:
        C_diffed += expTerm(k) * sinTerm(k) * integralTerm(k)

    # return C_diffed[1:-1]*2.0/(r[1:-1]*a)
    return C_diffed * 2.0 / (r * a)

def CJDiffuse_uniform(meanC,int_tao,r,a):
    '''

    THIS FUNCTINO IN PROGRESS, NOT YET WORKING

    Carlslaw and Jaeger solution for diffusion in a sphere of uniform concentration.

    In chapter 9.3 solution is listed as 'subtraction of 4 - 9 ' from V
    :param meanC: single value, the concentration
    :param int_tao: the integrated diffusivity, effectively k*t/a**2
    :param r: array of coordinates along the radius
    :param a:single value, radius of sphere
    :return: C(r,int_tao), the concentration after diffusion
    '''

    # kt = int_tao*a**2
    # ierfc = lambda z: (1.0/np.pi)*np.exp(-z**2) - z*special.erfc(z)

    #Solution to equation 4/5, zero concentration, surface temperature of V is constant
    ns = np.arange(1,1e3 + 1)

    eqn_5 = 0
    for n in ns:
        # eqn_5 = special.erfc(((2.0*n+1)*a-r)/(2.0*np.sqrt(kt))) - special.erfc(((2.0*n+1)*a+r)/(2.0*np.sqrt(kt)))
        eqn_5 += ((-1) ** n / n) * np.sin(n * np.pi * r / a) * np.exp(-(n ** 2) * (np.pi ** 2) * int_tao)

    # eqn_5 *= (a*meanC/r)
    eqn_5 = meanC + (2.0 * a * meanC / (np.pi * r)) * eqn_5

    #Solution to equation 6/7, temperature at center of sphere
    eqn_7 = (-1)**n*np.exp(-(n ** 2) * (np.pi ** 2) * int_tao)
    eqn_7 = meanC + 2.0*meanC*np.sum(eqn_7)

    #Solution to equation 8/9, average temperature
    eqn_9 = (1.0/n**2)*np.exp(-(n ** 2) * (np.pi ** 2) * int_tao)
    eqn_9 = meanC - (6.0*meanC/np.pi**2)*np.sum(eqn_9)


    # eqn_7 = np.exp(-(2.0*n+1)**2*a**2/(4.0*kt))
    # eqn_7 = np.sum(eqn_7)*a*meanC/np.sqrt(np.pi*kt)

    # eqn_9 = ierfc((n*a)/np.sqrt(kt))
    # eqn_9 = (6.0*meanC*np.sqrt(kt)/(a*np.sqrt(np.pi))) - (3.0*meanC*kt/a**2) + (12.0*meanC*np.sqrt(kt)/a)*np.sum(eqn_9)

    return meanC - (eqn_5 + eqn_7 + eqn_9)

def calcD_forArrhenius(a,sumF,dt):
    '''
    Calculates the Diffusivity for a degassing experiment based on equations
    for diffusion in a sphere of Fechtig and Kalbitzer, 1966

    Worth noting that
    :param a: Radius of grain (diffusion domain...)
    :param sumF:  Cumulative gas fraction released
    :param dt: duration of each step
    :return: D_i+1, the diffusivity
    '''

    #The first experiment is assumed to have a square profile, which requires its own equation (Eqns. 4a-4c)
    if sumF[0] <= 0.1:
        #Equation 4a
        D_0 = sumF[0]**2*np.pi*a**2/(36.0*dt[0])
    elif (sumF[0] > 0.1) & (sumF[0] < 0.1):
        # Equation 4b
        D_0 = (a**2/np.pi**2*dt[0])*(2.0*np.pi - np.pi**2*sumF[0]/3.0 - 2.0*np.pi*np.sqrt(1.0 - np.pi*sumF[0]/3.0))
    elif sumF[0] > 0.9:
        # Equation 4c
        D_0 = -(a**2/np.pi**2*dt[0])*np.log(np.pi**2*(1.0 - sumF[0])/6.0)



            #Subsequent experiments (e.g. not the first) have non-square profiles
    # due to diffusion in the sphere during the experiment (Eqns. 5a - 5b)
    Dip1_a = (sumF[1:] ** 2 - sumF[:-1] ** 2) * np.pi * a ** 2 / (36.0 * dt[1:])

    Dip1_b = (a ** 2 / (np.pi ** 2 * dt[1:])) * (-(np.pi ** 2 / 3.0) * (sumF[1:] - sumF[:-1])
                        - 2.0 * np.pi * (np.sqrt(1.0 - np.pi * sumF[1:] / 3.0) - np.sqrt(1.0 - np.pi * sumF[:-1] / 3.0)))

    Dip1_c = (a ** 2 / (np.pi ** 2 * dt[1:])) * np.log((1.0 - sumF[:-1]) / (1.0 - sumF[1:]))

    Dip1 = np.zeros_like(Dip1_a)
    Dip1[sumF[1:] <= 0.1] = Dip1_a[sumF[1:] <= 0.1]
    Dip1[(sumF[1:] > 0.1)] = Dip1_b[(sumF[1:] > 0.1)]
    Dip1[sumF[1:] > 0.9] = Dip1_c[sumF[1:] > 0.9]

    return np.hstack((D_0,Dip1))

def plotArrhenius(a,F3He,T,dt, **kwargs):
    '''

    :param kwargs:
    :return:
    '''

    D = calcD_forArrhenius(a,np.cumsum(F3He),dt)

    invT = 1 / T

    plt.plot(invT, np.log(D / a ** 2), **kwargs)

    return D,invT

def calcClosureTemperature(a,coolingRate,Ea,D_0,shape = 'sphere',closureTempGuess = 300.0):
    '''
    Calculate the closure temperature for the given cooling rate, given the constant Ea, and D_0
    describing thermally activated volume diffusion.
    :param a: spherical radius of grain (must match units of diffusion)
    :param coolingRate: degrees K / (time - units in D_0)
    :param Ea: Activation energy - J/Mol
    :param D_0: pre-exponential constant, units of length ^2 / time
    :param shape: determines 'A' the shape factor, options are 'sphere', 'cylinder', or 'sheet'
    :return:
    '''

    if shape.lower() == 'sphere':
        A = 55.0
    elif shape.lower() == 'cylinder':
        A = 27.0
    elif shape.lower() == 'plane sheet':
        A = 8.7

    #The most obvious strategy is to use the equations Dodson lays out (specifically this is from Equation 23
    # in Dodson 1973. Note that this is just the inverse (inverted) of what you'd expect from combi equtions i and ii in the abstract
    # this is the inverse of that.... For whatever reason, I get conversion with this version, but not when I try to just invert it
    # also, Dodson seems to use a negative for his dT/dt (my 'coolingRate' equivalent), which changes the sign.
    omega = A*D_0/a**2
    Tc_fun = lambda Tc: ((R_CONST / Ea) * np.log(omega * (R_CONST * Tc ** 2) / (Ea * coolingRate))) ** -1
    rootFun = lambda Tc: Tc - Tc_fun(Tc)

    #Another strategy would be to take the function for cooling rate provided in Reiners and Brandon, 2006 (Eqn 7),
    #and subtract the predicted cooling rate from the cooling rate. If the closure temperature is 'correct' this will
    #be zero - so we can search for this 'root' with newtons method
    # coolingRateFun = lambda Tc: (omega * R_CONST * Tc ** 2 / Ea) * np.exp(-Ea / (R_CONST * Tc))
    # rootFun = lambda Tc: coolingRate - coolingRateFun(Tc)


    Tc = optimize.newton(rootFun, closureTempGuess, maxiter=1000, tol=1e-6)

    return Tc - 273.15


# ==============================================================================
#  Classes for different thermochronometers
# ==============================================================================

class Thermochronometer():
    '''
    '''

    def __init__(self):
        self._parents = None
        self._daughters = None
        self._decayConsts = None
        self._daughterProductionFactors = None
        self._diffusivityFunction = None  # Afunction that excepts a teperature and returns a diffusivity
        self._multipleParents = None  # is there more than one parent
        pass

    def calcAge(self):

        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.

        If there are multiple parents, this is done via iterative, root finding methods.
        If there is a single parent the age is calculated directly.


        NOTE: Should add some checks to make sure that we don't try and calculate anything without
        first defining values
        '''
        if self._multipleParents:

            return self._calcAgeMultipleParents()

        else:

            return self._calcAgeSingleParent()

    def _calcAgeMultipleParents(self, t0=1e6):

        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.

        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods

        '''

        def rootFunc(self, t):
            sumParentsRemaining = np.zeros_like(self._daughter)
            sumDecayComponent = np.zeros_like(self._daughter)
            for i, f in enumerate(self._daughterProductionFactors):
                sumParentsRemaining += f * self._parents[i]
                sumDecayComponent += f * self._parents[i] * np.exp(self._decayConsts[i] * t)

            return (self._daughters + sumParentsRemaining) - sumDecayComponent

        tErr = lambda t: rootFunc(self, t)
        t0 = 1e6 * np.ones_like(self._daughters)  # Initial guess of 1 MA

        return optimize.root(tErr, t0).x

    def _calcIntegratedAgeMultipleParents(self, t0=1e6):

        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.

        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods

        '''

        def rootFunc(self, t):
            sumParentsRemaining = np.zeros_like(self._daughters)
            sumDecayComponent = np.zeros_like(self._daughters)
            for i, f in enumerate(self._daughterProductionFactors):
                sumParentsRemaining += f * np.sum(self._parents[i])
                sumDecayComponent += f * np.sum(self._parents[i] * np.exp(self._decayConsts[i] * t))

            return (self._daughters + sumParentsRemaining) - sumDecayComponent

        tErr = lambda t: rootFunc(self, t)
        t0 = 1e6 * np.ones_like(self._daughters)  # Initial guess of 1 MA

        return optimize.root(tErr, t0).x

    def _calcAgeSingleParent(self):

        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.

        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods

        '''

        return (1 / self._decayConsts) * np.log(1 + self._daughters / (self._daughterProductionFactors * self._parents))

    def _calcIntegratedAgeSingleParent(self):

        '''Calculate the raw age in years of the thermochronometer based on parent, daughter ratios.

        Given multiple parents, this is done via iterative methods. t0 is the initial
        guess for these methods

        '''

        return (1 / self._decayConsts) * np.log(
            1 + np.sum(self._daughters) / (self._daughterProductionFactors * np.sum(self._parents)))

    def calcDecayProductionRate(self):
        '''Rate of atoms of a parent lost for
        the decay constant lam and the number of parent atoms nParents

        dN/dt = -lam*N
        dD/dt = f*lam*N (where f is the number of daughters produced per parent decay)

        RETURNS:

        dN/dt - parent loss rate (negative)
        dD/dt -daughter accumulation rate (positive)
        '''

        if self._multipleParents:
            dNdt, dDdt = self._multiDecayProductionRate()
        else:
            dNdt = -self._decayConsts * self._parents
            dDdt = -self._daughterProductionFactors * dNdt

        return dNdt, dDdt

    def calcDaughterLossRate(self, T):

        ''' Calculates the rate of loss of daughter product at the specified temperature.
        The actual function for this will be defined by subclasses. All thermochronometers
        have a temperature dependent loss of daughter product, but they do not all
        share the same depdenencies.
        '''
        print('Error: calcDaughterLossRate not defined by subclass')
        return None

    def _multiDecayProductionRate(self):
        ''' For when multiple parents produce the same daughter, this calculates
        the rates of parent loss and daughter gain

        NOTE: SHOULD THINK ABOUT HOW I AM PASSING AND STORING THE PARENT CONC
        AND THE PRODUCTION RATE OF THAT CONC

        '''
        dDdt = np.zeros_like(self._parents[0])  # Preallocate space for the daughter production rate
        dNdt = []  # initialize a list for the parent decay rates

        for i, f in enumerate(self._daughterProductionFactors):
            dN_idt = -self._decayConsts[i] * self._parents[i]

            dD_idt = -f * dN_idt

            dDdt += dD_idt
            dNdt.append(dN_idt)

        return np.array(dNdt), dDdt


class sphericalThermochronometer(Thermochronometer):
    ''' A Basic spherical thermochronometer
    '''

    def __init__(self, radius, dr, diffusivityFunction, parentConcs, daughterConcs, decayConstants,
                 daughterProductionFactors):
        '''

        :param radius:
        :param dr:
        :param diffusivityFunction:
        :param parentConcs:
        :param daughterConcs:
        :param decayConstants:
        :param daughterProductionFactors:
        '''

        self.radius = radius
        self.dr = dr
        self.rs = np.arange(dr / 2.0, radius, dr)
        self._n = len(self.rs)
        self._diffusivityFunction = diffusivityFunction

        self._daughters = daughterConcs
        self._parents = parentConcs

        self._decayConsts = decayConstants
        self._daughterProductionFactors = daughterProductionFactors
        self.external_bc = 0.0

        # Set up matrices for solution
        a = np.array([np.ones(self._n - 1), np.ones(self._n), np.ones(self._n - 1)])

        # Initialize the matrices usef for the integration
        self._M = np.diag(np.ones(self._n), k=0)
        self._N = np.diag(np.ones(self._n), k=0)

        # Store the relative indices of the diagonals
        self._km1Diag = kth_diag_indices(self._M, -1)
        self._kDiag = np.diag_indices(self._n)
        self._kp1Diag = kth_diag_indices(self._M, 1)

        # fill in the otherimportant diagonals
        self._M[self._km1Diag] = 1.0
        self._M[self._kp1Diag] = 1.0

        self._N[self._km1Diag] = -1.0
        self._N[self._kp1Diag] = -1.0

        # Calculate the depletion fraction as a function of radius
        self._ejectionFraction = self._calcEjectionFraction()

        self._multipleParents = parentConcs.ndim > 1

        #What are the discrete volumes of each node spherical shell
        Volumes = (4.0 / 3.0) * np.pi * ((self.rs + self.dr / 2.0) ** 3 - (self.rs - (self.dr / 2.0)) ** 3)
        Volumes[0] = (4.0 / 3.0) * np.pi * (self.rs[0] + (self.dr / 2.0)) ** 3
        Volumes[-1] = (4.0 / 3.0) * np.pi * (self.radius ** 3 - (self.rs[-1] - (self.dr / 2.0)) ** 3)

        self._ShellVolumes = Volumes

    def _determineTimeStep(self, tNow, tStop, tempFromTimeFun, f):
        '''
        Function to determine the largest timestep that results in negligable changes in temperature across integration
        :param tNow:
        :param tStop:
        :param tempFromTimeFun:
        :return:
        '''

        def rootfun(dt):
            diffFun = lambda t: tempFromTimeFun(t)
            return np.abs(diffFun(tNow) - diffFun(tNow + dt)) - f

        # Split this up into log bins
        dts = np.logspace(0, 7, 100)

        # Calculate how close this is to the acceptable fraction
        errs = rootfun(dts)

        # Find the first unacceptable fraction
        idx = np.argwhere(errs < 0)[-1]
        dt = dts[idx]

        return dt

    def integrateThermalHistory_variableStep(self, tStart, tStop, tempFromTimeFun, f=5):
        '''
        Integrate through the provided thermal history, allowing the time step to vary to minimize misfit b/w temps b/w diffusivities
        :param tStart: time to start the integration
        :param tStop: time to end the integration
        :param tempFromTimeFun: function that given a time will return a temperature in K
        :param f: the target misfit between diffusivities at the start and end of timesteps
        :return: None, update the values of this instance
        '''

        t = tStart

        while t < tStop:
            dt = self._determineTimeStep(t, tStop, tempFromTimeFun, f)
            if t + dt > tStop:
                dt = tStop - t
            self.integrateTimestep(tempFromTimeFun(t + (dt / 2.0)), dt)
            t += dt

    def integrateThermalHistory(self, tStart, tStop, dt, tempFromTimeFun):
        '''
        Integrate through the provided thermal history
        :param tStart: time to start the integration
        :param tStop: time to stop the integration
        :param tempFromTimeFun: function that given a time will return a temperature in K
        :return: None, updates the values of this instance
        '''

        t = tStart

        while t < tStop:
            if t + dt > tStop:
                dt = tStop - t
            self.integrateTimestep(tempFromTimeFun(t + (dt / 2.0)), dt)
            t += dt

    def _calcEjectionFraction(self):
        '''
        No ejection for this thermochronometer
        :return:
        '''

        np.ones_like(self.rs)

    def calcDaughterLossRate(self, T):
        ''' Calculates the flux of daughter product following the temperature dependent
        diffusivity specified by the temperature T and the function of temperature
        diffusivityFunction. E.g.

        dNdT = D_0*exp(-Ea/RT) * grad^2 N
        '''

        # Apply boundary conditions - reflective and 0 concentration (outside grain)
        N = np.hstack((self._daughters[1], self._daughters, self.external_bc))

        grad2N = (N[2:] - 2.0 * N[1:-1] + N[:-2]) / (self.dr ** 2)

        return self._diffusivityFunction(T) * grad2N

    def integrateTimestep(self, T, dt):
        ''' Integrates the production diffusion equation
        '''
        dNdt, dDdt = self.calcDecayProductionRate()
        dDdt *= self._ejectionFraction
        self._parents += dNdt * dt  # Could make this more sophisticated

        #Integrate the production and diffution of the daughter
        self._daughters = self._integrateDiffusant(self._daughters,dDdt,T,dt)

    def plotDaughterProfile(self, normalize=True, **kwargs):
        '''

        :param kwargs: passed to matplotlib.plot(radialDistance,DaughterConc,**kwargs)
        :return:
        '''

        if normalize:
            plt.plot(self.rs / self.radius, self._daughters / np.max(self._daughters), '-', **kwargs)
        else:
            plt.plot(self.rs, self._daughters, '-', **kwargs)

    def _integrateDiffusant(self,diffusant,Production,T,dt):
        '''
        Integrate the production/diffusion equation in a sphere following the finite-difference solution of
        Ketcham 2005
        :param diffusant: The concentrations of the thing being diffused (e.g. the Daughter isotope)
        :param Production: The rate of production of the daughter isotope
        :param T: The temperature that is foverning diffusion
        :param dt: The timestep to integrate for
        :return:
        '''

        # Set up linear algebra solution
        # D = (self._diffusivityFunction(T_i) + self._diffusivityFunction(T_ip1))/2.0
        D = self._diffusivityFunction(T)
        b = 2.0 * self.dr ** 2 / (D * dt)
        self._M[self._kDiag] = (-b - 2.0)
        self._N[self._kDiag] = (2.0 - b)
        A = Production * self.rs * b * dt

        sum_RHS = (np.dot(self._N, diffusant * self.rs)) - A

        # Set boundary condition for external node
        sum_RHS[-1] = diffusant[-1] * self.rs[-1]
        self._M[-1, :] = 0
        self._M[-1, -1] = -1

        # Set boundary condition for central node
        sum_RHS[0] = (2 - b) * diffusant[0] * self.rs[0] - diffusant[1] * self.rs[1] + diffusant[0] * self.rs[0] - A[0]
        self._M[0, 0] = (-b - 3.0)

        return np.dot(sum_RHS, linalg.inv(self._M)) / self.rs #Was using this

        # return linalg.solve(self._M,sum_RHS)/ self.rs #But this seems more elegant, potentially more stable if the matrix is weird?


    def _volumeIntegral(self,quantity):
        '''

        :param quantity: the radial profile of some quantity
        :return: integral of the quantity over the spherical domain
        '''

        #We want to ingerate from 0 -> R, but our grid doesn't span that. Extend
        # and pad with zeros (first 0 is because concentration doesn't matter at r = 0, second is because concentration
        # is 0 at grain boundary)
        integralQuantity = np.hstack((0,quantity,0))
        positions = np.hstack((0,self.rs,self.radius))
        return sphericalVolumeIntegral(integralQuantity,positions)

class SphericalHeThermochronometer(sphericalThermochronometer):
    ''' A thermochronometer where the daughter product is He produced by decay of U, Th, Sm
    and daughter is lost by thermally activated diffusion
    '''
    stoppingDistance = None

    def _calcEjectionFraction(self):
        '''
        Eject He based on the stopping distance and grain size, utilizes the formulation of Farley for a spherical grain
        as summarized by Ketcham (2005)
        :return:
        '''

        Xstr = (self.rs ** 2 + self.radius ** 2 - self.stoppingDistance ** 2) / (2.0 * self.rs)
        ret = 0.5 + (Xstr - self.rs) / (2.0 * self.stoppingDistance)
        ret[(self.radius - self.stoppingDistance) > self.rs] = 1.0
        return ret

    def integrate43experiment(self,Temps = None,Durations = None,taos = None,plotProfileEvolution = False):
        '''

        :param Temps: The temperatures of each step
        :param Durations: The durations of each step
        :return: He_released, Rstep/Rbulk
        '''

        nPtsForDegassing = 500.0 #In my experience the fourier part of this solution performs best with a certiain number of pts

        if (Temps is None) or (Durations is None):
            Temps = np.linspace(130.0, 650.0, 100) + 273.15
            Durations = np.ones_like(Temps) * 0.1 / (24.0 * 365)

        if plotProfileEvolution:
            f,axs = plt.subplots(1,2)
            cmap = cm.get_cmap('coolwarm')
            colors = [cmap(i) for i in np.linspace(0,1,len(Temps))]

        #Creating copies of the 'doped' profile and the daughter profile
        #To forward diffuse

        #If using ketcham solution for diffusion, we have an irregular grid - interpolate to a regular grid
        dr = self.radius/nPtsForDegassing
        rs = np.arange(dr/1e2,self.radius,dr) #Make the first point very small, but non-zero (to avoid divide by zero in CJ diffusion)
        interp = interpolate.interp1d(np.hstack((-self.rs,self.rs,self.radius)),np.hstack((self._daughters,self._daughters,0)),kind = 'cubic')
        conc_4 = interp(rs)

        conc_3 = np.ones_like(conc_4)*np.mean(conc_4)

        if plotProfileEvolution:
            axs[0].plot(rs/self.radius,conc_3,'-k')
            axs[1].plot(rs/self.radius,conc_4,'-k')

        #Calculate how much gas started in the grains
        total3_0 = sphericalVolumeIntegral(conc_3,rs)
        total4_0 = sphericalVolumeIntegral(conc_4,rs)

        #If the diffusion parameters for each step where not specified, calculate them
        if taos is None:
            taos = self._diffusivityFunction(Temps) * Durations / self.radius ** 2

        #Preallocate some space for the results of step heating
        f_4He = np.zeros(len(taos)+1)
        f_3He = np.zeros(len(taos)+1)

        #f_0 = 0 (to start, all the gas is remaining) ... but Shuster seems to write in another way... that f_0 = 1? perhaps he means F_0 = 1?
        # f_4He[0] = 1.0
        # f_3He[0] = 1.0

        for i,tao in enumerate(taos):
            conc_3_i = CJDiffuse(conc_3,np.sum(taos[:i+1]),rs,self.radius)
            conc_4_i = CJDiffuse(conc_4,np.sum(taos[:i+1]),rs,self.radius)

            #For non-mirrored profile
            totalHe3_i = sphericalVolumeIntegral(conc_3_i,rs)
            totalHe4_i = sphericalVolumeIntegral(conc_4_i,rs)

            #First step is i=1, becuase at i = 0, f = 1 (all gas remaining),
            #Calculate the remaining gas fraction normalized by the initial gas content
            f_3He[i+1] = (total3_0 - totalHe3_i)/total3_0
            f_4He[i+1] = (total4_0 - totalHe4_i)/total4_0

            if plotProfileEvolution:
                axs[0].plot(rs[1:]/self.radius,conc_3_i[1:],label = str(i),color = colors[i])
                axs[1].plot(rs[1:]/self.radius,conc_4_i[1:],label = str(i),color = colors[i])

        if plotProfileEvolution:
            axs[0].set_ylabel(r'$[3^He]$',fontsize = 14)
            axs[0].set_xlabel(r'$r/a$',fontsize = 14)
            axs[1].set_ylabel(r'$[4^He]$', fontsize=14)
            axs[1].set_xlabel(r'$r/a$', fontsize=14)

        #Calculate the gas release fraction - Note: Shuster & Farley, 2004 have this written as f_(i-1) - f(i), which seems like the wrong sign to me...
        F3He = f_3He[1:] - f_3He[:-1]

        F4He = f_4He[1:] - f_4He[:-1]

        # This is written as Rstep in Shuster & Farley 2004, but is already normalized by the bulk ratio through f.
        #Where we to divide through be the bulk concentration ratio we would re-introduce dependence on concentration
        RstepRbulk = F4He/F3He
        return f_3He,f_4He,F3He,RstepRbulk


class SphericalApatiteHeThermochronometer(SphericalHeThermochronometer):
    '''
    A spherical he thermochronometer with the properties of apatite
    '''
    # Parameters related to decay
    _lam_238U = 1.55125E-10
    _lam_235U = 9.8485E-10
    _lam_232Th = 4.9475E-11
    _lam_147Sm = 6.539E-12

    # Production factors
    _p_238U = 8.0
    _p_235U = 7.0
    _p_232Th = 6.0
    _p_147Sm = 1.0

    _decayConsts = np.array([_lam_238U, _lam_235U, _lam_232Th])
    _daughterProductionFactors = np.array([_p_238U, _p_235U, _p_232Th])

    stoppingDistance = np.sum((np.array([19.68, 22.83, 22.46]) / 1e6) * _daughterProductionFactors) / np.sum(
        _daughterProductionFactors)

    def __init__(self, radius, dr, parentConcs, daughterConcs, diffusivityParams='Farley'):
        '''

        :param radius:
        :param dr:
        :param diffusivityFunction:
        :param parentConcs:
        :param daughterConcs:
        :param decayConstants:
        :param daughterProductionFactors:
        '''

        self.radius = radius
        self.dr = dr
        self.rs = np.arange(dr / 2.0, radius, dr) #Hmmm.... in some cases this can generate a radial coordinate AT the edge - which shouldn't exist given Ketcham's model setup...
        if self.rs[-1] == radius:
            print('Error: crystal grid coordinates do not abide by requirements of numerical model')

        self._n = len(self.rs)
        self._assignDiffusivityFunction(diffusivityParams)

        self._daughters = daughterConcs
        self._parents = parentConcs

        # Set up matrices for solution
        a = np.array([np.ones(self._n - 1), np.ones(self._n), np.ones(self._n - 1)])

        # Initialize the matrices usef for the integration
        self._M = np.diag(np.ones(self._n), k=0)
        self._N = np.diag(np.ones(self._n), k=0)

        # Store the relative indices of the diagonals
        self._km1Diag = kth_diag_indices(self._M, -1)
        self._kDiag = np.diag_indices(self._n)
        self._kp1Diag = kth_diag_indices(self._M, 1)

        # fill in the otherimportant diagonals
        self._M[self._km1Diag] = 1.0
        self._M[self._kp1Diag] = 1.0

        self._N[self._km1Diag] = -1.0
        self._N[self._kp1Diag] = -1.0

        # Calculate the depletion fraction as a function of radius
        self._ejectionFraction = self._calcEjectionFraction()

        self._multipleParents = parentConcs.ndim > 1

        #What are the discrete volumes of each node spherical shell
        Volumes = (4.0 / 3.0) * np.pi * ((self.rs + self.dr / 2.0) ** 3 - (self.rs - (self.dr / 2.0)) ** 3)
        Volumes[0] = (4.0 / 3.0) * np.pi * (self.rs[0] + (self.dr / 2.0)) ** 3
        Volumes[-1] = (4.0 / 3.0) * np.pi * (self.radius ** 3 - (self.rs[-1] - (self.dr / 2.0)) ** 3)

        self._ShellVolumes = Volumes

    def _assignDiffusivityFunction(self, diffusivityParams):
        '''

        :param diffusivityParams:
        :return:
        '''

        R = 8.3144598  # J/K /mol Universal gas constant

        if isinstance(diffusivityParams,dict):
            Do = diffusivityParams['Do']
            Ea = diffusivityParams['Ea']

        elif diffusivityParams is 'Cherniak':

            ##thermal diffusivty - should probably build all this into a file and just read from that, easier to update, switch between values
            # These values from Cherniak, Watson, and Thomas, 2009
            Do = 2.1E-6 * (60.0 * 60.0 * 24 * 365.0)  # m^2/s -> m^2 / yr
            Ea = (-117.0) * 1000.0  # kJ/mol -> J/mol

        elif diffusivityParams is 'Farley':
            # These values from Farley 2000
            Do = 157680.0  # m^2 / yr
            Ea = -137653.6  # J/mol

        self._diffusivityFunction = lambda T: thermDiffusivity(T, Do, Ea, R)


    def calcAge(self, applyFt=True):
        '''
        :param applyFt: boolean, should the alpha ejection correction be applied
        :return: age
        '''
        #
        # Volumes = (4.0 / 3.0) * np.pi * ((self.rs + self.dr / 2.0) ** 3 - (self.rs - (self.dr / 2.0)) ** 3)
        # Volumes[0] = (4.0 / 3.0) * np.pi * (self.rs[0] + (self.dr / 2.0)) ** 3
        # Volumes[-1] = (4.0 / 3.0) * np.pi * (self.radius ** 3 - (self.rs[-1] - (self.dr / 2.0)) ** 3)
        #
        # parents = [np.sum(parent * Volumes) for parent in self._parents]
        # daughters = np.sum(self._daughters * Volumes)

        parents = [self._volumeIntegral(parent) for parent in self._parents]
        daughters = self._volumeIntegral(self._daughters)

        def rootFunc(self, t):
            sumParentsRemaining = 0.0
            sumDecayComponent = 0.0
            for i, f in enumerate(self._daughterProductionFactors):
                sumParentsRemaining += f * parents[i]
                sumDecayComponent += f * parents[i] * np.exp(self._decayConsts[i] * t)

            return (daughters + sumParentsRemaining) - sumDecayComponent

        tErr = lambda t: rootFunc(self, t)
        t0 = 1e6  # Initial guess of 1 MA

        age = optimize.root(tErr, t0).x

        if applyFt:
            age /= self.calcFt()

        return age / 1e6

    def calcFt(self):
        '''
        Calculate the alpha ejection correction of Farley, 1996
        :return: Ft, the fraction of alpha particles that would be retained
        '''

        return 1 - (3.0 * self.stoppingDistance) / (4.0 * self.radius) + self.stoppingDistance ** 3 / (
        16.0 * self.radius ** 3)

# ==============================================================================
#  Classes for different thermochronology experiments
# ==============================================================================

class HeDegassingExperiment:
    '''
    This is a work in progress to experiment with inverting Apatite 4/3 data for thermal histories.
    '''

    def __init__(self,filepath,fitRange = [0,1],name = 'No name'):
        '''
        NOTES:
        :param filepath:
        :param fitIndices: Which range of sumFHe values might you want to fit? This is here in the even we want to ignore
        strange steps with little gas at the end of runs
        '''
        self.filepath = filepath
        self.fitRange = fitRange
        self.name = name

        #The expected structure of the excel files (from Shuster) have two headers, get informatino from the top first
        df = pd.read_excel(filepath,nrows = 2)

        self.R = float(df['grain_radius (cm)'][0] / 100.0) #cm -> m
        self.U = float(df['bulk[U] (ppm) '][0])
        self.Th = float(df['bulk[Th] (ppm)'][0])
        self.Age = float(df['age'][0])
        self.dAge = float(df['delage'][0])

        #Close the dataframe and re-open it skipping rows
        df = None
        df = pd.read_excel(filepath,skiprows=2)

        self.stepTemps = np.array(df['T(deg C) '])
        self.stepDurations = np.array(df['t(hr) '])/(24.0*365) # hrs -> years

        self.He_3 = np.array(df['He-3(10^6 atoms) '])
        self.dHe_3 =  np.array(df['dHe-3(10^6 atoms) '])

        self.HeRatio = np.array(df['4/3_ratio '])
        self.dHeRatio = np.array(df['d4/3_ratio '])

        self._goodData = (self.He_3 > 0) & (self.dHe_3 > 0) & (self.HeRatio > 0) & (self.dHeRatio > 0)
        self._goodIdcs = np.argwhere(self._goodData).flatten()

        #Hmmmm..... how to deal with bad experiments... tricky part about nan or zero is propogation of results
        self.HeRatio[~self._goodData] = 0
        self.dHeRatio[~self._goodData] = 0
        self.He_3[~self._goodData] = 0
        self.dHe_3[~self._goodData] = 0

        self.He_4 = self.He_3*self.HeRatio
        self.dHe_4 = self.He_4*np.sqrt((self.dHeRatio/self.HeRatio)**2 + (self.dHe_3/self.He_3)**2)
        self.dHe_4[~self._goodData] = 0

        self.F_3He, self.dF_3He, self.sum_F_3He, self.dsum_F_3He = self._calcf_F(self.He_3,self.dHe_3)

        self.fitIndices = (self.sum_F_3He >= fitRange[0]) & (self.sum_F_3He <=fitRange[1])

        self.RsRb = self.HeRatio/(np.sum(self.He_4)/np.sum(self.He_3))
        self.RsRb[~self._goodData] = 0

        dSum4He = np.sqrt(np.sum(self.dHe_4**2))
        dSum3He = np.sqrt(np.sum(self.dHe_3**2))
        Rb = np.sum(self.He_4)/np.sum(self.He_3)
        dRb = Rb*np.sqrt((dSum4He/np.sum(self.He_4))**2 + (dSum3He/np.sum(self.He_3))**2)
        self.dRsRb = self.RsRb*np.sqrt((self.dHeRatio/self.HeRatio)**2 + (dRb/Rb)**2)
        self.dRsRb[~self._goodData] = 0
        
        #Calculate the values of Tao estimated from this experiment - this is used for comparring this experiment to models
        self._Taos = self.calcTaosFromGasRelease()


    def _calcf_F(self,N,dN):
        '''
        :param N:
        :return: f
        '''
        N_0 = np.sum(N)
        dN_0 = np.sqrt(np.sum(dN**2))

        F = N/N_0
        dF = F*np.sqrt((dN_0/N_0)**2 + (dN/N)**2)

        F[~self._goodData] = 0
        dF[~self._goodData] = 0

        sum_F = np.cumsum(F)
        dSum_F = np.sqrt(np.cumsum(dF**2))

        sum_F[~self._goodData] = 0
        dSum_F[~self._goodData] = 0

        return F,dF,sum_F,dSum_F

    def plotRatioEvolution(self,**kwargs):
        '''
        :param kwargs: passed to pyplot.errorbar
        :return:
        '''
        plt.errorbar(self.sum_F_3He, self.RsRb, yerr=self.dRsRb, xerr=self.dsum_F_3He,label = self.name, **kwargs)

    def calcArrhenius(self):
        '''

        :param kwargs:
        :return:
        '''
        return calcD_forArrhenius(self.R,self.sum_F_3He[self._goodData],self.stepDurations[self._goodData]), 1.0/(273.15+self.stepTemps[self._goodData])

    def L2Norm_ratioEvolution(self,sumF3He_exp,RsRb):
        goodData = ~np.isnan(RsRb)  & self._goodData
        return np.sum((RsRb[goodData] - self.RsRb[goodData])**2)

    def ln_likelihood_RsRb(self,RsRb):
        goodData = ~np.isnan(RsRb) & self._goodData & self.fitIndices
        ps = (1.0/np.sqrt(2.0*np.pi*self.dRsRb**2))*np.exp(-(RsRb - self.dRsRb)**2/(2.0*self.dRsRb**2))
        return np.sum(np.log(ps[goodData]))

    def calcTaosFromGasRelease(self):
        '''
        Calculate the integrated normalized diffusivity determined from the gas release fractions based on the equations
        of  C&J. This will be used when comparing model to fit data, as it ensures that the x-axis are matched.
        :return: tao - an array of D*dt/a^2 for each step in the experiment
        '''
        
        D = self.calcArrhenius()[0] #Calc arrhenius masks out data that is not 'good', this needs to match
        tao = D*self.stepDurations[self._goodData]/self.R**2
        return tao

    def ln_likelihood_Experiment(self,HeModel):
        '''

        :param HeModel:
        :return:
        '''
        
        #To ensure matching gas release to experiment - use the tao values determined for this experiment in simulating the degassing        
        F3He, RsRb = HeModel.integrate43experiment(self.stepTemps + 273.15, self.stepDurations,taos= self._Taos,plotProfileEvolution=False)[2:]

        sumF3He = np.cumsum(F3He)

        #When fitting a more finely discretized step heating experiment
        goodIdcs = np.argwhere(self._goodData)

        return np.sum(stats.norm.logpdf(RsRb[self.fitIndices],self.RsRb[self.fitIndices],self.dRsRb[self.fitIndices]))

