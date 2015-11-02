'''
This code is used to generate random number

using Monte Carlo Simulation and multivariate normal distribution

implemented by weili zhang, 2014-10-23
'''

import numpy as np
import random

import matplotlib.pyplot as plt



def MNDNgenerator(mean, cov):
    '''
    generate random numbers following multivariate normal distribution
    '''
    random_vector = np.random.multivariate_normal(mean,cov)
    
    return random_vector


if __name__ == "__main__":

    mean = [0.6,0.6]
    cov = [[0.2,0.25],
           [0.25,0.2]]
    
    x = np.random.multivariate_normal(mean,cov)
    print x