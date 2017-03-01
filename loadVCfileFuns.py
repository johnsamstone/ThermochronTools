# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:59:06 2017

@author: sjohnstone


Functions for loading volume calibratino data
"""

import numpy as np
import VolumeCalibration as VC
import thermalTranspirationCorrection as ttc
from matplotlib import pyplot as plt

class tempGetter:
    
    def __init__(self):
        pass
    
    def getTemps(self,x):
        ''' Returns the temperature of the line and the temperature of the measurement
        '''
        return 20.55,45.0

def SJdataLoader(filenames,calibrationVolume,tempGetter,pipeDiameter,gas,expansionDirection):
    '''Function to transform the data stored in filename to a series of experiments
        File is tab delimited wth columns :  Pressure (mTorr), time (sec), Experiment number, isP1 (boolean)   
        
        In a miraculous display of trying to be clever, my 'p1' here refers to expanded data (p0) would be initial data. 
        Always good to try and be clever in the face of well accepted precedents
        
    '''
    
    #Get a thermal transpiration correction
    thermTransCorr = ttc.transpirationCorrection(Gas = gas, d= pipeDiameter)
    
    allExperiments = []      

    
    for filename in filenames:
    
        #Load in the data
        allData = np.loadtxt(filename,delimiter = '\t',skiprows = 1)
            
        allPs = allData[:,0]
        allts = allData[:,1]
        allExpNums = allData[:,2]
        all_isP2 = allData[:,3] == 1
        
        unqExps = np.unique(allExpNums)
      
        
        for unqExp in unqExps:
        
            #Find the indices corresponding to these data
            idcs = allExpNums == unqExp
            
            #Find the indices corresponding to the initial and expanded data
            initialData = idcs & ~all_isP2
            expandedData = idcs & all_isP2
            
            #Get the current temp
            Tline,Tgauge = tempGetter.getTemps(np.mean(allts[idcs]))       
            
            #Initialize a measurement class with these data
            P1s = allPs[initialData]
            P2s = allPs[expandedData]        
            
            #Assign the error to these measuremnets
            dP1s = P1s*0.012
            dP2s = P2s*0.012
            
            initialMeasurement = VC.pressureMeasurement(P1s,dP1s,Tline,Tgauge,thermTransCorr)
            expandedMeasurement = VC.pressureMeasurement(P2s,dP2s,Tline,Tgauge,thermTransCorr)
    
            #Use these to create an expansion experiment
            thisExperiment = VC.expansionExperiment(initialMeasurement,expandedMeasurement,thermTransCorr,calibrationVolume,expansionDirection)
            allExperiments.append(thisExperiment)    
    
            #print('Vu = %.2f +- %.3f %%'%(thisExperiment.Vu,100.0*thisExperiment.dVu/thisExperiment.Vu))

    expSet = VC.expansionExperimentSet(allExperiments)    
    
    return expSet
    
    
def getPressureError(pressure):
    
    
    dP = np.sqrt(((0.5e-6)**2 + ((2.5e-4)*pressure)**2 + ((1e-6)*pressure)**2 + (2e-7)**2))
    return dP
    
def LMdataLoader(filename,calibrationVolume,pipDiameter,gas):


    #This signify the closing of the valve    
    criticalVoltage = 4.275
    voltageTolerance = 0.0001
    
    #Load the data
    pressures,pipVs = LMimportData(filename)
    
    #Iterate through the data
    idxCntr = 0; 
    IDs = np.zeros_like(pressures) #Create a variable to indicate the different expansions  
    
    i = -1
    while i < len(pressures)-1:
        
        i+=1
        if (pipVs[i] > criticalVoltage - voltageTolerance) & (pipVs[i] < criticalVoltage + voltageTolerance):
            idxCntr+=1
        while (pipVs[i] > criticalVoltage - voltageTolerance) & (pipVs[i] < criticalVoltage + voltageTolerance):
            IDs[i] = idxCntr
            i+=1
            
    ##Find the unique values
    idcs = np.unique(IDs[IDs != 0])
    allMeas = []
    for idx in idcs:
        theseData = IDs==idx
        thisMeas = VC.pressureMeasurement(pressures[theseData],getPressureError(pressures[theseData]),45.0,45.0,ttc.transpirationCorrection('N2',0.01))
        allMeas.append(thisMeas)

    return allMeas

def LMimportData(filename):
    
    nSkipLines = 5
        
    Pressures = []
    PipVs = []
    
    with open(filename,'r') as f:
        for i,line in enumerate(f):
            if i >= nSkipLines:
                p,v = LMdataReadline(line)
                Pressures.append(p)
                PipVs.append(v)
    
    return np.array(Pressures),np.array(PipVs)
        
        

def LMdataReadline(line):
    vals = line.split('\t')
    Pressure = float(vals[10])
    PipV = float(vals[13])
    return Pressure,PipV