# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:31:41 2017

@author: sjohnstone

Class to store means and values for standard error propogation

"""
import numpy as np


class uncertainValue:
    def __init__(self, mean, error):
        self.mean = mean
        self.error = error

    def relErr(self):
        '''

        :return: dX / X
        '''

        return self.error/self.mean

    def __add__(self, other):
        '''
        this + that
        :param other:
        :return:
        '''

        #If the value to add doesn't have an uncertainty, assign it a zero uncertainty
        if not(isinstance(other,uncertainValue)):
            mean = self.mean + other
            error = self.error

        else:

            mean = self.mean + other.mean
            error = self._sumUncertainty(other)

        return uncertainValue(mean, error)

    def __sub__(self, other):
        '''

        this - that

        :param other:
        :return:
        '''

        # If the value to add doesn't have an uncertainty, assign it a zero uncertainty
        if not (isinstance(other, uncertainValue)):
            mean = self.mean - other
            error = self.error

        else:

            mean = self.mean - other.mean
            error = self._sumUncertainty(other)

        return uncertainValue(mean,error)


    def __div__(self, other):
        '''

        this / that

        :param other:
        :return:
        '''

        # If the value to add doesn't have an uncertainty, assign it a zero uncertainty
        if not (isinstance(other, uncertainValue)):
            mean = self.mean/other
            error = self.error/np.abs(other)

        else:

            mean = self.mean/other.mean
            error = self._multDivUncertainty(other,mean)

        return uncertainValue(mean,error)

    def __mul__(self, other):
        '''

        this * that

        :param other:
        :return:
        '''


        # If the value to add doesn't have an uncertainty, assign it a zero uncertainty
        if not (isinstance(other, uncertainValue)):
            mean = self.mean*other
            error = np.abs(other)*self.error

        else:

            mean = self.mean*other.mean
            error = self._multDivUncertainty(other,mean)

        return uncertainValue(mean,error)


    def __pow__(self, power, modulo=None):
        '''

        :param power:
        :param modulo:
        :return:
        '''

        mean = self.mean**power
        error = mean*power*(self.error/self.mean)

        return uncertainValue(mean,error)

    def _multDivUncertainty(self,other,newMean):

        return newMean*np.sqrt((self.error/self.mean)**2 + (other.error/other.mean)**2)

    def _sumUncertainty(self,other):

        return np.sqrt(self.error**2 + other.error**2)