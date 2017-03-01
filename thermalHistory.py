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
from matplotlib import cm
from scipy import optimize


class thermalHistory():

    ''' A Class that stores temperature-time information
    '''    
    
    t = None
    dt = None
    T = None
    name = None 
    
    def _init_():
        pass 

    def getTemp(self,time):
        ''' Return the temperature interpolated at the time specified 
        '''
        return np.interp(self.t,self.T,time)
        
    def plot(self,**kwargs):
        '''Plot the thermal history'''
        if self.name is None:
            plt.plot(self.t,self.T,**kwargs)
        else:
            plt.plot(self.t,self.T,label = self.name,**kwargs)
            
        plt.xlabel('Time (Ma)',fontsize = 13)
        plt.ylabel('Temperature (C)',fontsize = 13)
        
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
            
    