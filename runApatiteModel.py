import numpy as np
from matplotlib import pyplot as plt
from apatiteCrystalModel import *
from matplotlib import cm


#%%
#==============================================================================
# Actual model
#==============================================================================

dt = 500.0 #Times step in Yr )
Tmax = 30.0 *1E6
ts = np.arange(0,Tmax,dt)

Radius = 100.0 /1e6 #Crystal radius in m
dx = 1.0 / 1e6 #spacing of nodes in m
L = np.arange(0,Radius,dx) #Radial coordinates
nx = len(L)

#Concentrations
Conc238 = 8.0
Conc235 = (Conc238/137.0)
Conc232 = 147.0

#Parameters related to decay
lam_238U = 1.55125E-10
lam_235U = 9.8485E-10 
lam_232Th = 4.9475E-11 
lam_147Sm = 6.539E-12 

p_238U = 8.0
p_235U = 7.0
p_232Th = 6.0
p_147Sm = 1.0

decayConsts = (lam_238U, lam_235U, lam_232Th)
productionFactors = (p_238U, p_235U,p_232Th)

##redistribution function
stoppingDists = np.array([19.68, 22.83, 22.46])/1e6 #Stopping distances in microns

ejectionFunction = lambda D,i: alphaRedistribution(D,L,Radius,stoppingDists[i])

##thermal diffusivty
#These values from Cherniak, Watson, and Thomas, 2000
Do = 2.1E-6 *(60.0 *60.0*24*365.0) # m^2/s -> m^2 / yr
Ea = (-140.0)*1000.0 #kJ/mol -> J/mol
R = 8.3144598 #J/K /mol Universal gas constant 

diffusivity = lambda T: thermDiffusivity(T,Do,Ea,R)

#IF we want to compar a range of diffusivities, what is a reasonable range
#These parameters taken from Shuster et al, 2003, Farley, 2000, and Cherniak et al 2009

#%% Look at some whole grain profiles

#Total range
DosToTest = np.logspace(np.log10(2.1E-06),np.log(4.7E-02),5)*(60.0 *60.0*24*365.0) # m^2/s -> m^2 / yr
EasToTest = -np.linspace(111.0,148.0,5)*1000.0 #kJ/mol -> J/mol

#Different studies
DosToTest = np.array([2.1E-6,4.7E-02])*(60.0 *60.0*24*365.0) # m^2/s -> m^2 / yr
EasToTest = -np.array([111.0,148.0])*1000.0 #kJ/mol -> J/mol


getTemp = lambda t: 273.3+20.0

cmap = cm.get_cmap('plasma')
colors = [cmap(x) for x in np.linspace(0,1,len(DosToTest))]
    
for c in range(len(DosToTest)):
    #Temperature time path
    print('Modelling a new diffusivity')
    Ea = EasToTest[c]
    Do = DosToTest[c]
    #Define a new temperature dependent diffusivity
    diffusivity = lambda T: thermDiffusivity(T,Do,Ea,R)
    
    parentConcs = np.array([Conc238*np.ones_like(L), Conc235*np.ones_like(L), Conc232*np.ones_like(L)])
    daughterConcs = np.zeros_like(L)    
    
    for i,t in enumerate(ts):
        
        #Calculate rates of change
        dNdt,dDdT = multiDecayRate(parentConcs,decayConsts,productionFactors,redistributionFunction=ejectionFunction)
        for j in range(len(decayConsts)):    
            parentConcs+=dNdt[j]*dt
    
        #Apply boundary conditions - reflective and 0 concentration (outside grain)
        daughterConcs_bc = np.hstack((daughterConcs[1],daughterConcs,[0.0]))
        
        daughterDiff = diffusion(daughterConcs_bc,dx,getTemp(t),diffusivity)
        daughterConcs+=(dDdT + daughterDiff)*dt
        
#        if np.mod(i,500)==0:
#            plt.figure(1)
#            plt.clf()
#            plt.plot(L*1e6,daughterConcs,'or')
#            plt.title(' t =  %.0f yrs'%(t/(np.pi*1E7)))
#            plt.show()
#            plt.pause(0.1)
    
    plt.figure(2)
    ages = calcAge(parentConcs,daughterConcs,decayConsts,productionFactors)
    plt.plot(L*1e6,ages/1e6,'-o',linewidth = 2, color = colors[c], label = 'Ea = %.1f, Do = %.1E'%(Ea/1000.0,Do/(60.0 *60.0*24*365.0)))
    plt.pause(0.5)        
    plt.show()
    
plt.legend(loc = 'lower left')    
plt.ylabel('Age (Ma)',fontsize = 13)
plt.xlabel('Distance from grain center (um)')
plt.title('T = %.0f K'%getTemp(0))


#%% Look at some varying grain sizes

RadiiToTest = np.logspace(np.log10(25),np.log10(300),10)/1e6

#Total range
DosToTest = np.logspace(np.log10(2.1E-06),np.log(4.7E-02),7)*(60.0 *60.0*24*365.0) # m^2/s -> m^2 / yr
EasToTest = -np.linspace(117.0,148.0,7)*1000.0 #kJ/mol -> J/mol
EasToTest = EasToTest[::-1] #Flip so that effects are more pronounced in same direction as Dos

#Different studies
#DosToTest = np.array([2.1E-6,4.7E-02])*(60.0 *60.0*24*365.0) # m^2/s -> m^2 / yr
#EasToTest = -np.array([111.0,148.0])*1000.0 #kJ/mol -> J/mol


getTemp = lambda t: 273.3+20.0

cmap = cm.get_cmap('plasma')
colors = [cmap(x) for x in np.linspace(0,1,len(DosToTest))]
 
for i,r in enumerate(RadiiToTest):

    L = np.arange(0,r,dx) #Radial coordinates
    nx = len(L) 
    ejectionFunction = lambda D,i: alphaRedistribution(D,L,r,stoppingDists[i])
    
    for c in range(len(DosToTest)):
        #Temperature time path
        print('Modelling a new diffusivity')
        Ea = EasToTest[c]
        Do = DosToTest[c]
        #Define a new temperature dependent diffusivity
        diffusivity = lambda T: thermDiffusivity(T,Do,Ea,R)
        
        parentConcs = np.array([Conc238*np.ones_like(L), Conc235*np.ones_like(L), Conc232*np.ones_like(L)])
        daughterConcs = np.zeros_like(L)    
        
        for t in (ts):
            
            #Calculate rates of change
            dNdt,dDdT = multiDecayRate(parentConcs,decayConsts,productionFactors,redistributionFunction=ejectionFunction)
            for j in range(len(decayConsts)):    
                parentConcs+=dNdt[j]*dt
        
            #Apply boundary conditions - reflective and 0 concentration (outside grain)
            daughterConcs_bc = np.hstack((daughterConcs[1],daughterConcs,[0.0]))
            
            daughterDiff = diffusion(daughterConcs_bc,dx,getTemp(t),diffusivity)
            daughterConcs+=(dDdT + daughterDiff)*dt
            
        
        plt.figure(2)
        totalD = np.sum(daughterConcs)
        totalP = [np.sum(parentConcs[i]) for i in range(len(productionFactors))]
        ages = calcAge(totalP,[totalD],decayConsts,productionFactors)
        

        plt.plot(r*1e6,ages/1e6,'o', color = colors[c], label = 'Ea = %.1f, Do = %.1E'%(Ea/1000.0,Do/(60.0 *60.0*24*365.0)))

        plt.pause(0.5)        
        plt.show()
    
plt.legend(loc = 'lower right')    
plt.ylabel('Age (Ma)',fontsize = 13)
plt.xlabel('Grain radius (um)')
plt.title('T = %.0f K'%getTemp(0))
