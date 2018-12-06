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
Conc238 = 50.0#8.0
Conc235 = (Conc238 / 137.0)
Conc232 = 65.0#147.0

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

#These step heating experiments from a set of experiments from Shuster
stepHeatTemps = np.array([180.0,225,260,300,300,310,330,340,350,350,370,400,410,420,440,475,500,600,700,900])+273.15
stepHeatDurations = np.array([1.0,0.5,0.38,0.51,0.66,0.66,0.46,0.45,0.48,0.66,0.53,0.48,0.50,0.56,0.63,0.5,0.5,0.5,0.5,0.5])/(24.0*365.0) #half an hour each, converted to years

stepHeatTemps = stepHeatTemps[1:]
stepHeatDurations = stepHeatDurations[1:]

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



#########################################################################################################
#### Fifth test,Shuster and Farley ratio evolution diagram- multiple histories
#########################################################################################################

#Default params
Radius = 65.0 / 1e6  # Crystal radius in m
dx = 1.0 / 1e6  # spacing of nodes in m
L =  np.arange(dx / 2.0, Radius, dx)

# Concentrations
Conc238 = 1.0#8.0
Conc235 = (Conc238 / 137.0)
Conc232 = 1.0#147.0

parentConcs = np.array([Conc238 * np.ones_like(L), Conc235 * np.ones_like(L), Conc232 * np.ones_like(L)])
daughterConcs = np.zeros_like(L)


diffusivity = 'Farley'

thermalHistories = []
colors = []

#Instantaneous quenching at 10 ma
timePoints = np.array([10.0, 0.0])*1e6
thermalPoints = np.array([5.0, 5.0])+273.15
thermalHistories.append(tHist.thermalHistory(-timePoints,thermalPoints))
colors.append('orangered')

#Constant cooling rate
timePoints = np.array([15.0, 0.0])*1e6
thermalPoints = np.array([88.0, 25.0])+273.15
thermalHistories.append(tHist.thermalHistory(-timePoints,thermalPoints))
colors.append('green')

#Holding, then quenching
timePoints = np.array([15.0, 5.0,5.01,0.0])*1e6
thermalPoints = np.array([64.0, 64.0, 5.0, 5.0])+273.15
thermalHistories.append(tHist.thermalHistory(-timePoints,thermalPoints))
colors.append('k')

#Constant temp, steady state profile
timePoints = np.array([100.0, 0.0])*1e6
thermalPoints = np.array([63.0, 63.0])+273.15
thermalHistories.append(tHist.thermalHistory(-timePoints,thermalPoints))
colors.append('lavender')

#These step heating experiments from a set of experiments from Shuster
stepHeatTemps = np.array([180.0,225,260,300,300,310,330,340,350,350,370,400,410,420,440,475,500,600,700,900])+273.15
stepHeatDurations = np.array([1.0,0.5,0.38,0.51,0.66,0.66,0.46,0.45,0.48,0.66,0.53,0.48,0.50,0.56,0.63,0.5,0.5,0.5,0.5,0.5])/(24.0*365.0) #half an hour each, converted to years

# stepHeatTemps = np.linspace(150.0,900.0,30)+273.15
# stepHeatDurations = np.ones_like(stepHeatTemps)*0.3 / (24.0*365)

f,axs = plt.subplots(3,1)
for i,thermalHistory in enumerate(thermalHistories):


    HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,np.copy(parentConcs),np.copy(daughterConcs),diffusivityParams=diffusivity)

    #Integrate this thermal history with a fixed timestemp
    HeModel.integrateThermalHistory(thermalHistory.t[0], thermalHistory.t[-1],1e5, thermalHistory.getTemp)

    axs[0].plot(-thermalHistory.t/1e6,thermalHistory.T - 273.15,'-',color = colors[i])
    axs[0].set_ylim(100.0,0)
    axs[0].set_xlim(15.0,0)
    axs[0].set_ylabel('Temperature C')
    axs[0].set_xlabel('Time (Ma)')

    plt.sca(axs[1])
    thisAge = HeModel.calcAge(applyFt=True)
    HeModel.plotDaughterProfile(normalize=False,linewidth=2, color=colors[i],label = 'He age = %.1f Ma'%thisAge)
    axs[1].set_ylim(0,1)
    axs[1].set_xlim(0,1)
    axs[1].grid()
    axs[1].set_ylabel(r'$[^4He]$',fontsize = 13)
    axs[1].set_xlabel('Normalized Radius')
    axs[1].legend(loc = 'best')

    f_3He,f_4He,F3He,RsRb = HeModel.integrate43experiment(stepHeatTemps,stepHeatDurations,plotProfileEvolution=False)

    axs[2].plot(np.cumsum(F3He),RsRb,'-',color = colors[i])
    axs[2].set_xlabel(r'$\sum F ^3He$',fontsize = 14)
    axs[2].set_ylabel(r'$R_{step}/R_{bulk}$',fontsize = 14)
    axs[2].set_ylim(0,1.6)

#########################################################################################################
#### Sixth test,Shuster and Farley ratio evolution diagram- for an arbitrary history (using this to compare with Hefty)
#########################################################################################################

#Default params
Radius = 65.0 / 1e6  # Crystal radius in m
dx = .10 / 1e6  # spacing of nodes in m
L =  np.arange(dx / 2.0, Radius, dx)

# Concentrations
Conc238 = 1.0#8.0
Conc235 = (Conc238 / 137.0)
Conc232 = 1.0#147.0

parentConcs = np.array([Conc238 * np.ones_like(L), Conc235 * np.ones_like(L), Conc232 * np.ones_like(L)])
daughterConcs = np.zeros_like(L)


diffusivity = 'Farley'

thermalHistories = []
colors = []

#Some cooling history
timePoints = np.array([30.0,20.0, 0.0])*1e6
thermalPoints = np.array([200.0,0.0, 0.0])+273.15
thermalHistory = tHist.thermalHistory(-timePoints,thermalPoints)
color = 'r'

#These step heating experiments from a set of experiments from Shuster
# stepHeatTemps = np.array([180.0,225,260,300,300,310,330,340,350,350,370,400,410,420,440,475,500,600,700,900])+273.15
# stepHeatDurations = np.array([1.0,0.5,0.38,0.51,0.66,0.66,0.46,0.45,0.48,0.66,0.53,0.48,0.50,0.56,0.63,0.5,0.5,0.5,0.5,0.5])/(24.0*365.0) #half an hour each, converted to years

stepHeatTemps = np.linspace(150.0,900.0,30)+273.15
stepHeatDurations = np.ones_like(stepHeatTemps)*0.3 / (24.0*365)

f,axs = plt.subplots(3,1)

HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,np.copy(parentConcs),np.copy(daughterConcs),diffusivityParams=diffusivity)

#Integrate this thermal history with a fixed timestemp
HeModel.integrateThermalHistory(thermalHistory.t[0], thermalHistory.t[-1],1e5, thermalHistory.getTemp)

axs[0].plot(-thermalHistory.t/1e6,thermalHistory.T - 273.15,'-',color = color)
axs[0].set_ylim(100.0,0)
axs[0].set_xlim(15.0,0)
axs[0].set_ylabel('Temperature C')
axs[0].set_xlabel('Time (Ma)')

plt.sca(axs[1])
thisAge = HeModel.calcAge(applyFt=True)
HeModel.plotDaughterProfile(normalize=True,linewidth=2, color=color,label = 'He age = %.1f Ma'%thisAge)
axs[1].set_ylim(0,1)
axs[1].set_xlim(0,1)
axs[1].grid()
axs[1].set_ylabel(r'$[^4He]$',fontsize = 13)
axs[1].set_xlabel('Normalized Radius')
axs[1].legend(loc = 'best')

f_3He,f_4He,F3He,RsRb = HeModel.integrate43experiment(stepHeatTemps,stepHeatDurations,plotProfileEvolution=False)

axs[2].plot(np.cumsum(F3He),RsRb,'-',color = color)
axs[2].set_xlabel(r'$\sum F ^3He$',fontsize = 14)
axs[2].set_ylabel(r'$R_{step}/R_{bulk}$',fontsize = 14)
axs[2].set_ylim(0,1.6)
axs[2].grid()


#########################################################################################################
#### Seventh test, Impact of diffusivity on fraction of gas released
#########################################################################################################

#Default params
Radius = 65.0 / 1e6  # Crystal radius in m
dx = 1.0 / 1e6  # spacing of nodes in m
L =  np.arange(dx / 2.0, Radius, dx)

# Concentrations
Conc238 = 1.0#8.0
Conc235 = (Conc238 / 137.0)
Conc232 = 1.0#147.0

parentConcs = np.array([Conc238 * np.ones_like(L), Conc235 * np.ones_like(L), Conc232 * np.ones_like(L)])
daughterConcs = np.zeros_like(L)

diffusivities = ['Farley', 'Cherniak']
colors = ['orangered','dodgerblue']

#Some cooling history
timePoints = np.array([30.0,20.0, 0.0])*1e6
thermalPoints = np.array([200.0,0.0, 0.0])+273.15
thermalHistory = tHist.thermalHistory(-timePoints,thermalPoints)
color = 'r'

#These step heating experiments from a set of experiments from Shuster
# stepHeatTemps = np.array([180.0,225,260,300,300,310,330,340,350,350,370,400,410,420,440,475,500,600,700,900])+273.15
# stepHeatDurations = np.array([1.0,0.5,0.38,0.51,0.66,0.66,0.46,0.45,0.48,0.66,0.53,0.48,0.50,0.56,0.63,0.5,0.5,0.5,0.5,0.5])/(24.0*365.0) #half an hour each, converted to years

stepHeatTemps = np.linspace(150.0,900.0,30)+273.15
stepHeatDurations = np.ones_like(stepHeatTemps)*0.3 / (24.0*365)

f,axs = plt.subplots(3,1)
F3Hes = []

for i, diffusivity in enumerate(diffusivities):

    color = colors[i]

    HeModel = tchron.SphericalApatiteHeThermochronometer(Radius,dx,np.copy(parentConcs),np.copy(daughterConcs),diffusivityParams=diffusivity)

    #Integrate this thermal history with a fixed timestemp
    HeModel.integrateThermalHistory(thermalHistory.t[0], thermalHistory.t[-1],1e5, thermalHistory.getTemp)

    axs[0].plot(-thermalHistory.t/1e6,thermalHistory.T - 273.15,'-',color = color)
    axs[0].set_ylim(100.0,0)
    axs[0].set_xlim(15.0,0)
    axs[0].set_ylabel('Temperature C')
    axs[0].set_xlabel('Time (Ma)')


    f_3He,f_4He,F3He,RsRb = HeModel.integrate43experiment(stepHeatTemps,stepHeatDurations,plotProfileEvolution=False)

    axs[1].plot(np.cumsum(F3He),RsRb,'-x',color = color)
    axs[1].set_xlabel(r'$\sum F ^3He$',fontsize = 14)
    axs[1].set_ylabel(r'$R_{step}/R_{bulk}$',fontsize = 14)
    axs[1].set_ylim(0,1.6)
    axs[1].grid()

    F3Hes.append(F3He)

for i in range(1,len(F3Hes)):
    axs[2].plot(np.cumsum(F3Hes[0]),np.cumsum(F3Hes[i]),'o',color = colors[i],label = diffusivities[i])

axs[2].legend()
axs[2].set_xlabel('Gas released, {}'.format(diffusivities[i]))
axs[2].set_ylabel('Gas released')
axs[2].plot([0,1],[0,1],'--k')



#########################################################################################################
#### Eighth test,Inversion using Shuster and Farley ratio evolution diagram
#########################################################################################################

from scipy.optimize import minimize
dt = 1e5

#First create a function that creates a thermal history (flat isotherm then cooling from that)
def createThermalHistory(T_0,t_0,t_c):
    '''
    This is a cooling history that starts with isothermal cooling and then cools rapidly to the surface
    :param T_0: Initial temperature in C
    :param t_0: Ma, Starting time of cooling history (will probably leave this fixed as it may often be unconstrained)
    :param t_c: Ma, Starting time of cooling
    :return:
    '''

    timePoints = np.array([t_0,t_c,0]) * 1e6
    thermalPoints = np.array([T_0,T_0,0]) + 273.15

    return tHist.thermalHistory(-timePoints,thermalPoints)

def predictRatioEvolution(thermalHistory,stepHeatTemps,stepHeatDurations):
    HeModel = tchron.SphericalApatiteHeThermochronometer(Radius, dx, parentConcs, daughterConcs, diffusivityParams=diffusivity)
    HeModel.integrateThermalHistory(-timePoints[0],-timePoints[-1],dt,thermalHistory.getTemp)
    f_3He, f_4He, F3He, RsRb = HeModel.integrate43experiment(stepHeatTemps, stepHeatDurations,
                                                             plotProfileEvolution=False)
    return np.cumsum(F3He),RsRb

def L2norm(sumF3He_obs,RsRb_obs,T_0,t_0,t_c,stepHeatTemps,stepHeatDurations):
    thist = createThermalHistory(T_0,t_0,t_c)
    sumF3He_exp, RsRb_exp = predictRatioEvolution(thist,stepHeatTemps,stepHeatDurations)
    goodData = ~np.isnan(RsRb_exp) & ~np.isnan(RsRb_obs)
    return 100.0*np.sum((RsRb_obs[goodData] - RsRb_exp[goodData])**2) #Sum of squares alone is very small... this was quicker than adjusting tolerance

relNoise = 0.05
nIterations = 3
T_0_act = 50.0
t_0_act = 30.0
t_c_act = 10.0

initialGuesses = 5.0
bounds = [(0,500.0),(0,t_0_act)]

stepHeatTemps = np.linspace(280.0,575.0,20.0)+273.15
stepHeatDurations = np.ones_like(stepHeatTemps)*0.33/(24.0*365.0) #half an hour each, converted to years

thermalHistory_act = createThermalHistory(T_0_act,t_0_act,t_c_act)
sumF3He_act,RsRb_act = predictRatioEvolution(thermalHistory_act,stepHeatTemps,stepHeatDurations)

f,axs = plt.subplots(1,2)

for i in range(nIterations):
    RsRb_obs = RsRb_act + np.random.randn(len(RsRb_act))*relNoise*RsRb_act

    objFun = lambda params: L2norm(sumF3He_act,RsRb_obs,T_0_act,t_0_act,params,stepHeatTemps,stepHeatDurations)

    pred = minimize(objFun,initialGuesses,method='Nelder-Mead')#,bounds = bounds)
    tHist_pred = createThermalHistory(T_0_act,t_0_act,pred.x)

    axs[0].plot(sumF3He_act,RsRb_obs,'-k',alpha = 0.2)
    plt.sca(axs[1])
    tHist_pred.plot(color = 'k',alpha = 0.2)


    #Plot best fitting ratio evolution diagram
    sumF3He_pred,RsRb_pred = predictRatioEvolution(tHist_pred,stepHeatTemps,stepHeatDurations)
    axs[0].plot(sumF3He_pred,RsRb_pred,'-ob',alpha = 0.2)

#Plot actual data
axs[0].plot(sumF3He_act,RsRb_act,'-r',linewidth = 2)
plt.sca(axs[1])
thermalHistory_act.plot(color = 'r',linewidth = 2, label = 'actual')
plt.legend()

