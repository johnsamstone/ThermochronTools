# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:16:36 2017

@author: sjohnstone
"""

import numpy as np

#==============================================================================
# Definition of constants
#==============================================================================
R = 8.3144598 #Universal gas constant in J/(mol K)


class transpirationCorrection:
    
    
    An = None
    Bn = None
    Cn = None
    Gas = None
    d = None
    
    def __init__(self,Gas = 'N2',d = 0.0):
        self.Gas = Gas
        self.d = d 
        
        if Gas == 'He':
            self.An = 0.0318
            self.Bn = 0.252
            self.Cn = 0.176
            self.M = 4.003 #g/Mol
            self.eta = 1.985E-5 #Viscosity at 298.15 K, Pa s
            self.dEtadT = 4.47E-8 #Temperature dependence on viscosity, Pa s / mol
            
        elif Gas == 'N2':
            self.An = 0.0275
            self.Bn = 0.321
            self.Cn = 0.322
            self.M = 28.013 #g/Mol
            self.eta = 1.777E-5 #Viscosity at 298.15 K, Pa s
            self.dEtadT = 4.66E-8 #Temperature dependence on viscosity, Pa s / mol
            
        elif Gas == 'Ar':
            self.An = 0.0262
            self.Bn = 0.275
            self.Cn = 0.272 
            self.M = 39.948 #g/Mol
            self.eta = 2.263E-5 #Viscosity at 298.15 K, Pa s
            self.dEtadT = 6.65E-8 #Temperature dependence on viscosity, Pa s / mol            
            
        elif Gas == 'H2':
            self.An = 0.0318
            self.Bn = 0.319
            self.Cn = 0.180
            self.M = 2.016 #g/Mol
            self.eta = 8.9E-6 #Viscosity at 298.15 K, Pa s
            self.dEtadT = 2.1E-8 #Temperature dependence on viscosity, Pa s / mol            
            
        else:
            print('Error: Please select a known gas; N2, He, Ar, or H2')
            
        
        self.cFun = lambda T: np.sqrt(3.0*R*T/(self.M/1e3))
        
    def applyCorrection(self,p_meas,t_gauge,t_line):
        ''' Returns the corrected pressure based on the measured pressure, temperatures, 
        and the parameters of this correction
        '''
        c = self.cFun((t_gauge+t_line)/2.0) #Calculate the RMS gas velocity using the average temperature
        p_star = self.eta*c/self.d #P_star as defined in equation 4 of Setina, 1999 Metrologia
        psi = p_meas/p_star
        p2overp1 = ((self.An * psi**2 + self.Bn*psi + self.Cn*psi**0.5 + (t_gauge/t_line)**0.5)
                    /(self.An*psi**2 + self.Bn*psi + self.Cn*psi**0.5 + 1)) #equation 5 of Setina, 1999 Metrologia
        return p_meas/p2overp1