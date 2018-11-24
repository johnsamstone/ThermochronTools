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
from scipy import interpolate
from matplotlib import cm
from scipy import integrate


# ==============================================================================
# Constants used throughout
# ==============================================================================
R = 8.0  # Universal gas constant
Av = 6.022  # Avogadros number


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
    if np.mod(n,2)==0:
        n+=1
    Ks = np.arange(1, len(C))

    expTerm = lambda k: np.exp(-(k ** 2) * (np.pi ** 2) * int_tao)
    sinTerm = lambda k: np.sin(k * np.pi * r / a)
    integralTerm = lambda k: integrate.trapz(sinTerm(k) * r * C, r)

    C_diffed = np.zeros_like(C)

    for k in Ks:
        C_diffed += expTerm(k) * sinTerm(k) * integralTerm(k)

    # return C_diffed[1:-1]*2.0/(r[1:-1]*a)
    return C_diffed * 2.0 / (r * a)


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

        return np.dot(sum_RHS, linalg.inv(self._M)) / self.rs

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

    def integrate43experiment(self,Temps,Durations,plotProfileEvolution = False):
        '''

        :param Temps: The temperatures of each step
        :param Durations: The durations of each step
        :return: He_released, Rstep/Rbulk
        '''

        if plotProfileEvolution:
            f,axs = plt.subplots(1,2)
            cmap = cm.get_cmap('coolwarm')
            colors = [cmap(i) for i in np.linspace(0,1,len(Temps))]

        #Creating copies of the 'doped' profile and the daughter profile
        #To forward diffuse

        #If using ketcham solution for diffusion, we have an irregular grid - interpolate to a regular grid
        rs = np.arange(0,self.radius,self.dr) #Make the first point very small, but non-zero (to avoid divide by zero in CJ diffusion)
        interp = interpolate.interp1d(np.hstack((-self.rs,self.rs,self.radius)),np.hstack((self._daughters,self._daughters,0)),kind = 'cubic')
        conc_4 = interp(rs)

        # conc_4 = np.copy(self._daughters)
        # rs = self.rs

        conc_3 = np.ones_like(conc_4)*np.mean(conc_4)
        # conc_3[-1] = 0

        if plotProfileEvolution:
            axs[0].plot(rs/self.radius,conc_3,'-k')
            axs[1].plot(rs/self.radius,conc_4,'-k')

        #Calculate how much gas started in the grains
        total3_0 = sphericalVolumeIntegral(conc_3,rs)
        total4_0 = sphericalVolumeIntegral(conc_4,rs)

        #Preallocate some space for the results of step heating
        f_4He = np.zeros(len(Temps)+1)
        f_3He = np.zeros(len(Temps)+1)

        #f_0 = 0 (to start, all the gas is remaining) ... but Shuster seems to write in another way... that f_0 = 1? perhaps he means F_0 = 1?
        # f_4He[0] = 1.0
        # f_3He[0] = 1.0

        taos = self._diffusivityFunction(Temps)*Durations/self.radius**2

        for i,tao in enumerate(taos):
            conc_3_i = CJDiffuse(conc_3,np.sum(taos[:i+1]),rs,self.radius)
            conc_4_i = CJDiffuse(conc_4,np.sum(taos[:i+1]),rs,self.radius)

            conc_3_i[0] = 0
            conc_4_i[0] = 0

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

        #Calculate the gas release fraction - Note: Shuster & Farley have this written as f_(i-1) - f(i), which seems like the wrong sign to me...
        F3He = f_3He[1:] - f_3He[:-1]

        F4He = f_4He[1:] - f_4He[:-1]

        RBulk = total4_0/total3_0

        RstepRbulk = (F4He/F3He)/RBulk

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

    def __init__(self, radius, dr, parentConcs, daughterConcs, diffusivityParams='Cherniak'):
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

        if diffusivityParams is 'Cherniak':

            ##thermal diffusivty - should probably build all this into a file and just read from that, easier to update, switch between values
            # These values from Cherniak, Watson, and Thomas, 2000
            Do = 2.1E-6 * (60.0 * 60.0 * 24 * 365.0)  # m^2/s -> m^2 / yr
            Ea = (-140.0) * 1000.0  # kJ/mol -> J/mol
            R = 8.3144598  # J/K /mol Universal gas constant

        elif diffusivityParams is 'Farley':
            # These values from Farley 2000
            Do = 157680.0  # m^2 / yr
            Ea = -137653.6  # J/mol
            R = 8.3144598  # J/K /mol Universal gas constant

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