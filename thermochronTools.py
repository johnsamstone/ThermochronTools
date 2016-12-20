import numpy as np
from matplotlib import pyplot as plt



class collection(self):
    #A collection is a suite of samples 

    def __init__(self):
        pass
    
    def readFromFile(self,filename):
        
        with filename open as f:
            # Read the file
    
    def saveToFile(self,filename):
        
        with filename open as f:
            # Write the file
            
            # Write the header
            
            # Write the entries
            for sample in collection:
                
                

class sample(self):

    def __init__(self):
        self.lat
        self.lon
        self.name
        self.info
    
    def printEntry(self,formatter,seperator)
        #Returns a string constructed to match the style specified by formatter.
        
        #Get the fields that are to be used in the ouput
        fields = formatter.split(seperator)
        
        #Initialize output string
        textEntry = ''
        
        #Iterate through the fields and print each
        for f in fields:
            
            if hasattr(self,f):
                #Print field
                textEntry = textEntry + getattr(self,f) + seperator
            else 
                textEntry = textEntry + ' ' + seperator 
        
        return textEntry
    

class UThHe(sample):

    def __init__(self):
        self.U
        self.Th
        self.Sm
        self.He
        self.__IsCalcd = False
        self.grainDimensions

class AHe(UThHe):
    
    _Do_ = 
    _Ea_ = 
        
    
    def __init__(self):
        self.geometry


    def calcAge(self):
    
        self.__IsCalcd = True
        self.Age = 
    
    def calcFT(self):

        
        self.Ft =
        
    def getDiffusivity(self,T):
        
        return D = 

       
