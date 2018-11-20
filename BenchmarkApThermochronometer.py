'''
Benchmark the apatite He Thermochronometer to the solutions presented in Farley (2005) specifically those presented in
Figure 10
'''

import Thermochronometer as tchron
import numpy as np
import thermalHistory as tHist
from matplotlib import pyplot as plt


#Default params
Radius = 100.0 / 1e6  # Crystal radius in m
dx =1.0 / 1e6  # spacing of nodes in m
L =  np.arange(dx / 2.0, Radius, dx)
Vs = (4.0/3.0)*np.pi*((L+(dx/2))**3 - (L-(dx/2.0))**3)
weights = Vs/np.sum(Vs)
nx = len(L)

# Concentrations
Conc238 = 1.0#8.0
Conc235 = (Conc238 / 137.0)
Conc232 = 1.0#147.0

parentConcs = np.array([Conc238 * np.ones_like(L), Conc235 * np.ones_like(L), Conc232 * np.ones_like(L)])
daughterConcs = np.zeros_like(L)


diffusivity = 'Farley'

#Whats the greatest number of degrees allowable in a given time step
maxDegreeJump = 2
dt = 1e4


#########################################################################################################
#### First test, top row in figure 10
#########################################################################################################

timePoints = np.array([60.0, 50.0, 0.0])*1e6
thermalPoints = np.array([120.0, 20.0, 20.1])+273.15

HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,parentConcs,daughterConcs,diffusivityParams=diffusivity)
thermalHistory = tHist.thermalHistory(-timePoints,thermalPoints)

#Integrate this thermal history with a variable timestemp
HeModel.integrateThermalHistory_variableStep(-timePoints[0], timePoints[-1], thermalHistory.getTemp, f=maxDegreeJump)

#Integrate this thermal history with a fixed timestemp
# HeModel.integrateThermalHistory(-timePoints[0], timePoints[-1],dt, thermalHistory.getTemp)

plt.subplot(3,2,1)
plt.plot(timePoints/1e6,thermalPoints - 273.15,'-ok')
plt.ylim(200.0,0)
plt.xlim(100.0,0)
plt.grid()
plt.ylabel('Temperature C')


plt.subplot(3,2,2)
thisAge = HeModel.calcAge(applyFt=True)
HeModel.plotDaughterProfile(normalize=True,linewidth=2, color='k',label = 'He age = %.1f Ma'%thisAge)
plt.ylim(0,1)
plt.xlim(0,1)
plt.grid()
plt.ylabel('Normalized Concentration')
plt.legend(loc = 'best')



#########################################################################################################
#### Second test, middle row in figure 10
#########################################################################################################

timePoints = np.array([60.0, 0.0])*1e6
thermalPoints = np.array([120.0, 20.0])+273.15

HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,parentConcs,daughterConcs,diffusivityParams=diffusivity)
thermalHistory = tHist.thermalHistory(-timePoints,thermalPoints)

#Integrate this thermal history with a variable timestemp
HeModel.integrateThermalHistory_variableStep(-timePoints[0], timePoints[-1], thermalHistory.getTemp, f=maxDegreeJump)

#Integrate this thermal history with a fixed timestemp
# HeModel.integrateThermalHistory(-timePoints[0], timePoints[-1],dt, thermalHistory.getTemp)


plt.subplot(3,2,3)
plt.plot(timePoints/1e6,thermalPoints - 273.15,'-ok')
plt.ylim(200.0,0)
plt.xlim(100.0,0)
plt.grid()
plt.ylabel('Temperature C')


plt.subplot(3,2,4)
thisAge = HeModel.calcAge(applyFt=True)
HeModel.plotDaughterProfile(normalize=True,linewidth=2, color='k',label = 'He age = %.1f Ma'%thisAge)
plt.ylim(0,1)
plt.xlim(0,1)
plt.grid()
plt.ylabel('Normalized Concentration')
plt.legend(loc = 'best')

#########################################################################################################
#### Third test, bottom row in figure 10
#########################################################################################################

timePoints = np.array([60.0, 2.5, 0.0])*1e6
thermalPoints = np.array([120.0, 65.0, 20.0])+273.15

HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,parentConcs,daughterConcs,diffusivityParams=diffusivity)
thermalHistory = tHist.thermalHistory(-timePoints,thermalPoints)

#Integrate this thermal history with a variable timestemp
HeModel.integrateThermalHistory_variableStep(-timePoints[0], timePoints[-1], thermalHistory.getTemp, f=maxDegreeJump)

#Integrate this thermal history with a fixed timestemp
# HeModel.integrateThermalHistory(-timePoints[0], timePoints[-1],dt, thermalHistory.getTemp)

plt.subplot(3,2,5)
plt.plot(timePoints/1e6,thermalPoints - 273.15,'-ok')
plt.ylim(200.0,0)
plt.xlim(100.0,0)
plt.grid()
plt.ylabel('Temperature C')
plt.xlabel('Time (Ma)')

plt.subplot(3,2,6)
thisAge = HeModel.calcAge(applyFt=True)
HeModel.plotDaughterProfile(normalize=True,linewidth=2, color='k',label = 'He age = %.1f Ma'%thisAge)
plt.ylim(0,1)
plt.xlim(0,1)
plt.grid()
plt.ylabel('Normalized Concentration')
plt.xlabel('Normalized Radius')
plt.legend(loc = 'best')

#########################################################################################################
#### Fourth test,Shuster and Farley ratio evolution diagram
#########################################################################################################

timePoints = np.array([15.0, 0.0])*1e6
thermalPoints = np.array([90.0, 25.0])+273.15

stepHeatTemps = np.arange(240.0,600.0,20.0)+273.15
stepHeatTemps = np.hstack((stepHeatTemps))
stepHeatDurations = np.ones_like(stepHeatTemps)*0.5/(24.0*365.0) #half an hour each, converted to years

HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,parentConcs,daughterConcs,diffusivityParams=diffusivity)
thermalHistory = tHist.thermalHistory(-timePoints,thermalPoints)

#Integrate this thermal history with a variable timestemp
# HeModel.integrateThermalHistory_variableStep(-timePoints[0], timePoints[-1], thermalHistory.getTemp, f=maxDegreeJump)

#Integrate this thermal history with a fixed timestemp
HeModel.integrateThermalHistory(-timePoints[0], timePoints[-1],dt, thermalHistory.getTemp)

f,axs = plt.subplots(1,3)
axs[0].plot(timePoints/1e6,thermalPoints - 273.15,'-ok')
axs[0].set_ylim(200.0,0)
axs[0].set_xlim(100.0,0)
axs[0].grid()
axs[0].set_ylabel('Temperature C')
axs[0].set_xlabel('Time (Ma)')

plt.sca(axs[1])
thisAge = HeModel.calcAge(applyFt=True)
HeModel.plotDaughterProfile(normalize=True,linewidth=2, color='k',label = 'He age = %.1f Ma'%thisAge)
axs[1].set_ylim(0,1)
axs[1].set_xlim(0,1)
axs[1].grid()
axs[1].set_ylabel('Normalized Concentration')
axs[1].set_xlabel('Normalized Radius')
axs[1].legend(loc = 'best')

f_3He,f_4He,F3He,RsRb = HeModel.integrate43experiment(stepHeatTemps,stepHeatDurations,plotProfileEvolution=True)

axs[2].plot(np.cumsum(F3He),RsRb,'-ok')
axs[2].set_xlabel(r'$\sum F ^3He$',fontsize = 14)
axs[2].set_ylabel(r'$R_{step}/R_{bulk}$',fontsize = 14)