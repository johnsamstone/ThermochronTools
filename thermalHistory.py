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
from matplotlib import cm
from scipy import optimize
from scipy import interpolate


class thermalHistory:

    ''' A Class that stores temperature-time information
    '''    
    
    t = None
    dt = None
    T = None
    name = None 
    
    def __init__(self,t,T):
        self.t = t
        self.T = T

        self.getTemp = interpolate.interp1d(t,T,fill_value=(T[0],T[-1]),bounds_error=False)
        
    def plot(self,**kwargs):
        '''Plot the thermal history'''
        if self.name is None:
            plt.plot(self.t,self.T,**kwargs)
        else:
            plt.plot(self.t,self.T,label = self.name,**kwargs)
            
        plt.xlabel('Time (Ma)',fontsize = 13)
        plt.ylabel('Temperature (C)',fontsize = 13)

    def prependtT(self,t,T):
        '''
        Add this temp and time to the begenning of the time temp history
        :param t:
        :param T:
        :return:
        '''
        self.t = np.append(t,self.t)
        self.T = np.append(T,self.T)

        self.getTemp = interpolate.interp1d(self.t,self.T,fill_value=(self.T[0],self.T[-1]),bounds_error=False)

    def appendtT(self,t,T):
        '''
        Add this temp and time to the end of the time temp history
        :param t:
        :param T:
        :return:
        '''
        self.t = np.append(self.t,t)
        self.T = np.append(self.T,T)

        self.getTemp = interpolate.interp1d(self.t,self.T,fill_value=(self.T[0],self.T[-1]),bounds_error=False)

    def saveToFile(self,filePath):
        '''
        Save this thermal history to a comma delimited text file
        :param filePath:
        :return:
        '''

        formatCode = '%.1f, %.0f \n'
        with open(filePath,'w') as f:
            f.write('t, T\n')
            for t,T in zip(self.t,self.T):
                f.write(formatCode%(t,T))

class thermalTransect():
    ''' A Class that stores a series of thermal histories, intended to represent 
    a transect. 
    
    For example, in a perfectly vertical transect with a steady-state vertical transect,
    
    '''
    n = None #Number of samples
    thermalHistories = None #List of individual thermal histories
    def _init_():
        pass        
    
    def add(self,thermalHistory):
        ''' Add a thermal history to the list of thermal histories
        '''
        self.n+=1
        self.thermalHistories.append(thermalHistory)
        
    def plot(self,cmap = 'viridis',**kwargs):
        ''' Plot the thermal histories stored within this transect, color them according to
        the specified colormap
        '''
        cmap = cm.get_cmap(name=cmap)
        colors = [cmap(x) for x in np.linspace(0,1,self.n)]
        
        for i,thermalHistory in enumerate(self.thermalHistories):
            thermalHistory.plot(linecolor = colors[i])
            
    