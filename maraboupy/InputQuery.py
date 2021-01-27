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

import warnings
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
from maraboupy import Marabou
from maraboupy import MarabouNetwork
from maraboupy import Property


class InputQuery:

    def __init__(self):
        self.ipq = MarabouCore.InputQuery()
        self.network = MarabouNetwork.MarabouNetwork()
        self.num_vars = 0
        self.property = Property.Property(property_filename="")
        self.ipq_initialized = False
        self.network_initialized = False
        # self.marabou_options = MarabouCore.MarabouOptions()
        self.ipq_preprocessed = False


    def initializeFromNNet(self, network_filename: str, property_filename = ""):
        nnet = Marabou.read_nnet(filename=network_filename)
        self.initializeFromNetwork(marabou_network=nnet)


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

    def initializePropertyFromFile(self, property_filename='', update_ipq=True):
        self.property = Property.Property(property_filename=property_filename)
        if update_ipq:
            self.updateIPQFromProperty()

    def updateIPQFromProperty(self):
        # Updating bounds
        for (variable, indices, type_of_bound, bound) in self.property.marabou_bounds:
            var = self.computeIPQVarFromIndex(variable=variable, indices=indices)
            if type_of_bound == 'r':  # Upper bound
                self.setUpperBound(var=var, ub=bound)
            else:
                self.setLowerBound(var=var, lb=bound)

        # Adding equations
        for property_equation in self.property.marabou_equations:
            eq = self.createEquationFromPropertyEq(property_equation=property_equation)
            self.addEquation(e=eq)

    def createEquationFromPropertyEq(self, property_equation):
        """

        Args:
            property_equation:

        Returns:

        :meta private:
        """
        eq = MarabouUtils.Equation(EquationType=property_equation['eq_type'])
        eq.setScalar(x=property_equation['scalar'])

        # Adding addends
        for (coeff, variable, indices) in property_equation['addends']:
            var = self.computeIPQVarFromIndex(variable=variable, indices=indices)
            eq.addAddend(c=coeff, x=var)

        return eq

    def computeIPQVarFromIndex(self, variable: str, indices: list):
        """

        Args:
            variable:
            indices:

        Returns:

        :meta private:
        """
        if variable == 'x':  # Input bound
            assert indices[0] < self.ipq.getNumInputVariables()
            return self.ipq.inputVariableByIndex(indices[0])
        elif variable == 'y':  # Output bound
            assert indices[0] < self.ipq.getNumOutputVariables()
            return self.ipq.outputVariableByIndex(indices[0])
        else:
            # Hidden neuron. Currently only supported for the nnet case.
            # TODO: expose nlr to python and compute variable directly from nlr?
            return self.computeHiddenVariableForNNet(ipq_layer=indices[0], node=indices[1])

    def computeHiddenVariableForNNet(self, ipq_layer: int, node: int):
        """ Function to compute...

            Assumes that self.network is an MarabouNetworkNNet

        Args:
            layer (int):
            node (int):

        Returns:
            (int)

        :meta private:
        """
        # if ipq_layer == 0: # Input layer
        #     return node
        assert self.network_initialized
        try:
            layer = int((ipq_layer + 1) / 2)
            assert layer < self.network.numLayers

            if ipq_layer % 2 == 0:  # Even layer: relu (f)
                return self.network.nodeTo_f(layer=layer, node=node)
            else:
                return self.network.nodeTo_b(layer=layer, node=node)

        except NameError:
            warnings.warn("ipq.network is not an nnet! Can't compute a hidden variable index.")
            assert False
        except AssertionError:
            warnings.warn("Layer or index out of bounds! Can't compute a hidden variable index.")
            assert False


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

    def setOptions(self, options):
        self.marabou_options = options

    def preprocess(self):
        self.ipq = MarabouCore.preprocess(self.ipq, self.marabou_options)
        self.ipq_preprocessed = True

    def solve(self, preprocess=False):
        if preprocess and not self.ipq_preprocessed:
            self.preprocess()
        return MarabouCore.solve(self.ipq, self.marabou_options)
