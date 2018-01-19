# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28, 2017

@author: sjohnstone

Run script to compute the mols of gas added to a the USGS He line tanks

"""

import TankFillingTools as tft
from errorPropogation import uncertainValue as uV
from thermalTranspirationCorrection import transpirationCorrection as tC
import numpy as np

from matplotlib import pyplot as plt

transCorr = tC(Gas = 'He',d = 0.01,isTempCelsius=False)

#==============================================================================
# Tank 1, 2 Shots of 3 He
#==============================================================================

##Relevant volumes
V_p = uV(0.092362479, 0.000208646)/1e6 #Volume of pipette 1
V_t = uV(14892.9690097518,6.645024347)/1e6 #Volume of tank 1

#Filling data file path
filePath = '/Users/sjohnstone/Documents/HeLab/TankData/TankFilling/TankFill_Tank1_2ndAttempt_3He.txt'

expIDs = [6, 8] #ID numbers of expansions
TLines = np.array([21.2, 21.3])+273.15
TGauges = np.array([45.0, 45.0])+273.15

Tank1 = tft.fillingSeries(filePath,transCorr,expIDs, TGauges,TLines,V_t,V_p)

print('For tank 1, %.2e mols of 3 He +/- %.2e (%.3f %%)'%(Tank1.n_gas.mean,Tank1.n_gas.error, 100.0*Tank1.n_gas.relErr()))

#==============================================================================
# Tank 2, 2 Shots of 4 He
#==============================================================================

##Relevant volumes
V_p = uV(0.087506557,0.000125485)/1e6 #Volume of pipette 1
V_t = uV(14875.11461, 6.24022087)/1e6 #Volume of tank 1

#Filling data file path
filePath = '/Users/sjohnstone/Documents/HeLab/TankData/TankFilling/TankFill_Tank2_4He.txt'

expIDs = [1, 2] #ID numbers of expansions
TLines = np.array([20.9, 21.1])+273.15
TGauges = np.array([45.0, 45.0])+273.15

Tank2 = tft.fillingSeries(filePath,transCorr,expIDs, TGauges,TLines,V_t,V_p)

print('For tank 2, %.2e mols of 4 He +/- %.2e (%.3f %%)'%(Tank2.n_gas.mean,Tank2.n_gas.error, 100.0*Tank2.n_gas.relErr()))

#==============================================================================
# Tank 1, 2  Shots of 4 He
#==============================================================================

##Relevant volumes
V_p = uV(0.092362479,0.000208646)/1e6 #Volume of pipette 1
V_t = uV(14892.96901, 6.645024347)/1e6 #Volume of tank 1

#Filling data file path
filePath = '/Users/sjohnstone/Documents/HeLab/TankData/TankFilling/TankFill_Tank3_4He.txt'

expIDs = [1, 2] #ID numbers of expansions
TLines = np.array([20.8, 20.9])+273.15
TGauges = np.array([45.0, 45.0])+273.15

Tank3 = tft.fillingSeries(filePath,transCorr,expIDs, TGauges,TLines,V_t,V_p)

print('For tank 3, %.2e mols of 3 He +/- %.2e (%.3f %%)'%(Tank3.n_gas.mean,Tank3.n_gas.error, 100.0*Tank3.n_gas.relErr()))
