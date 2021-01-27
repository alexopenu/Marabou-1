'''
Top contributors (to current version):
    - Alex Usvyatsov

This file is part of the Marabou project.
Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.

Marabou defines key functions that make up the main user interface to Maraboupy
TODO: change
'''

# import warnings
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
# from maraboupy import Marabou
from maraboupy import MarabouNetwork



class InputQuery:

    def __init__(self):
        self.ipq = MarabouCore.InputQuery()
        self.network = MarabouNetwork.MarabouNetwork()
        self.num_vars = 0
        self.ipq_initialized = False
        self.network_initialized = False
        self.marabou_options = MarabouCore.MarabouOptions()
        self.ipq_preprocessed = False

    def initializeFromNetwork(self, marabou_network: MarabouNetwork.MarabouNetwork):
        self.network = marabou_network
        self.ipq = self.network.getMarabouQuery()
        self.network_initialized = True
        self.ipq_initialized = True
        self.num_vars = self.ipq.getNumberOfVariables()

    def initializeFromIPQ(self, ipq: MarabouCore.InputQuery):
        self.ipq = ipq
        self.ipq_initialized = True
        self.network_initialized = False
        self.num_vars = self.ipq.getNumberOfVariables()


    def addEquation(self, e: MarabouUtils.Equation):
        eq = MarabouCore.Equation(e.EquationType)

        for (c, var) in e.addendList:
            assert var < self.num_vars
            eq.addAddend(c, var)
        eq.setScalar(e.scalar)
        self.ipq.addEquation(eq)

    def setUpperBound(self, var: int, ub: float):
        assert var < self.num_vars
        self.ipq.setUpperBound(var, ub)

    def setLowerBound(self, var: int, lb: float):
        assert var < self.num_vars
        self.ipq.setLowerBound(var, lb)

    def getUpperBound(self, var: int):
        assert var < self.num_vars
        return self.ipq.getUpperBound(var)

    def getLowerBound(self, var: int):
        assert var < self.num_vars
        return self.ipq.getLowerBound(var)

    def setOptions(self, options: MarabouCore.MarabouOptions):
        self.marabou_options = options

    def preprocess(self):
        self.ipq = MarabouCore.preprocess(self.ipq, self.marabou_options)
        self.ipq_preprocessed = True

    def solve(self, preprocess = False):
        if preprocess and not self.ipq_preprocessed:
            self.preprocess()
        return MarabouCore.solve(self.ipq, self.marabou_options)


