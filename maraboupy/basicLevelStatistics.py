
'''
/* *******************                                                        */
/*! \file basicLevelStatistics.py
 ** \verbatim
 ** Top contributors (to current version):
 ** Alex Usvyatsov
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief
 ** This class computes basic statistics on inputs of neurons for a layer of a neural network
 ** based on a list of observed values
 ** More generally, given a matrix sample[i][var] where i runs over the samples, and var runs over
 ** the variables of the layer, computes the following statistics
 **     minimums:       list of observed minimums for the layer (so minimum[var] is the observed minimum for var)
 **     maximums:       same for maximums
 **     mean:           list of  means of observed values
 **     range:          list of observed ranges (so range[var] is the observed range for var)
 **     epsiloni, epsiloni_twosided: differences between observed minima/maxima to the predicted lower and upper bounds
 **         computed based on very simple statistics
 **         epsiloni is a list
 **         epsiloni_twosided is a dictionary
 **             the keys are 'l' (for 'left' or 'lower'), 'r' (for 'right' or upper)
 **             the values are lists
 **
 ** [[ Add lengthier description here ]]
 **/
'''



from MarabouNetworkNNetExtended import *

from Marabou import *
# from MarabouCore import *
# from MarabouNetworkNNetExtentions import *

# import re
# import parser

import sys
import time



# from random import choice
# from random import choices

import warnings

try:
    import seaborn as sns
except ImportError:
    warnings.warn('Module seaborn not installed')

import matplotlib.pyplot as plt
import numpy as np


class basicLevelStatistics:
    def __init__(self, sample_values_list: list, epsilon=0.01, two_sided = True):
        '''

        :param sample_values_list: list of sample inputs for the layer
        :param epsilon: an initial constant value for difference between observed and actual extreme values
        :param two_sided (bool): if True, statistics are computed separately for lower and upper bounds
        '''
        self.epsilon = epsilon

        self.layer_size = 0
        self.sample_matrix = np.array([])
        self.minimums = []
        self.maximums = []
        self.mean = []
        self.epsiloni = []
        self.epsiloni_twosided = {'l': [], 'r': []}
        self.range = []

        self.statistics_computed = False

        if len(sample_values_list):
            self.recomputeStatistics(sample_values_list, two_sided)


    def recomputeStatistics(self, sample_values_list: list, two_sided=True):
        '''
        Computes the actual statistics.
        :param sample_values_list: list of sample inputs for the layer
        :param two_sided (bool): if True, statistics are computed separately for lower and upper bounds
        :return: None
        '''

        if not len(sample_values_list):
            self.statistics_computed = False
            self.layer_size = 0
            self.sample_matrix = np.array([])
            self.minimums = []
            self.maximums = []
            self.mean = []
            self.epsiloni = []
            self.epsiloni_twosided = {'l': [], 'r': []}
            self.range = []
        else:
            self.sample_matrix = np.array(sample_values_list)
            self.layer_size = len(sample_values_list[0])
            self.minimums = [np.min(self.sample_matrix[:, var]) for var in range(self.layer_size)]
            self.maximums = [np.max(self.sample_matrix[:, var]) for var in range(self.layer_size)]
            self.mean = [np.mean(self.sample_matrix[:, var]) for var in range(self.layer_size)]
            self.range = [self.maximums[var]-self.minimums[var] for var in range(self.layer_size)]
            if two_sided:
                self.epsiloni_twosided = {'l': [0]*self.layer_size, 'r': [0]*self.layer_size}
            else:
                self.epsiloni_twosided = {'l': [], 'r': []}
            self.epsiloni = [0]*self.layer_size

            self.computeEpsilonsUniformly(two_sided = two_sided)
            self.statistics_computed = True



    def computeEpsilonsUniformly(self, two_sided = True):
        '''
        computes epsilons (to be added to or subtracted from empiric upper and lower bounds for the hidden layer
            variables) based on a very simple statistical heuristics, essentially assuming that the distribution
            on the values of each variable is close to uniform. This is not the case, but the computation is
            efficient, and seems to give "conservative" bounds, which should be a good starting point.
        :return: None
        '''

        assert self.sample_matrix.__len__()
        for var in range(self.layer_size):
            self.computeEpsilonsForVariable(var, compute_twosided=two_sided)


    def computeEpsilonsForVariable(self,var, compute_twosided = True):

        outputs = self.sample_matrix[:, var]
        sample_size = len(outputs)

        minimum = self.minimums[var]
        maximum = self.maximums[var]
        mean = self.mean[var]

        # self.range[var] = maximum-minimum
        self.epsiloni[var] = self.range[var]/sample_size

        if compute_twosided:
            small_outputs = [output for output in outputs if (output <= mean)]
            small_range = mean-minimum
            big_outputs = [output for output in outputs if (output > mean)]
            big_range = maximum-mean

            self.epsiloni_twosided['l'][var] = small_range/len(small_outputs)
            self.epsiloni_twosided['r'][var] = big_range/len(big_outputs)


    def graphVariableDist(self, var: int):

        assert self.statistics_computed
        assert var in range(self.layer_size)

        try:
            sns.distplot([output[var] for output in self.sample_matrix], label='Variable' + str(var))
        except NameError:
            warnings.warn('seaborn package has not been imported, can not plot the distribution.')
            return

        plt.legend()
        plt.show()


