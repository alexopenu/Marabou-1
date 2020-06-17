

'''
/* *******************                                                        */
/*! \file MarabouNetworkNNeIPQ.py
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
 ** This class extends MarabouNetworkNNet class
 ** Adds an Input Query object as an additional attribute
 ** Adds features that allow updating the the MarabouNetworkNNet object from the IPQ
 **
 ** [[ Add lengthier description here ]]
 **/
'''




import MarabouNetworkNNetExtendedParent
import MarabouCore
import numpy as np

class MarabouNetworkNNetIPQ(MarabouNetworkNNetExtendedParent.MarabouNetworkNNetExtendedParent):
    """
    Class that implements a MarabouNetwork from an NNet file.
    """
    def __init__ (self, filename="", property_filename = "", use_nlr = False, compute_ipq = False):
        """
        Constructs a MarabouNetworkNNetIPQ object from an .nnet file.
        Computes InputQuery, potentially in two ways
        ipq1 is computed from the MarabouNetworkNNet object itself
        ipq2 is computed directly from the nnet file

        Args:
            filename: path to the .nnet file.
            property_filename: path to the property file

        Attributes:
            ipq1             an Input Query object containing the Input Query corresponding to the network
            ipq2             an Input Query object created from the file (and maybe property file)

        Attributes from MarabouNetworkNNet:

            numLayers        (int) The number of layers in the network
            layerSizes       (list of ints) Layer sizes.
            inputSize        (int) Size of the input.
            outputSize       (int) Size of the output.
            maxLayersize     (int) Size of largest layer.
            inputMinimums    (list of floats) Minimum value for each input.
            inputMaximums    (list of floats) Maximum value for each input.
            inputMeans       (list of floats) Mean value for each input.
            inputRanges      (list of floats) Range for each input
            weights          (list of list of lists) Outer index corresponds to layer
                                number.
            biases           (list of lists) Outer index corresponds to layer number.
            sbt              The SymbolicBoundTightener object

            inputVars
            b_variables
            f_variables
            outputVars

        Attributes from MarabouNetwork

            self.numVars
            self.equList = []
            self.reluList = []
            self.maxList = []
            self.varsParticipatingInConstraints = set()
            self.lowerBounds = dict()
            self.upperBounds = dict()
            self.inputVars = []
            self.outputVars = np.array([])


        """
        super(MarabouNetworkNNetIPQ,self).__init__(filename=filename,property_filename=property_filename,use_nlr=use_nlr,compute_ipq=compute_ipq)
        if compute_ipq:
            self.ipq1 = self.getMarabouQuery()
        else:
            self.ipq1 = MarabouCore.InputQuery()
        self.ipq2 = self.getMarabouQuery()
        if filename:
            MarabouCore.createInputQuery(self.ipq2,filename,property_filename)


    def computeIPQ(self):
        self.ipq1 = self.getMarabouQuery()

    def getInputQuery(self,networkFilename,propertyFilename):
        MarabouCore.createInputQuery(self.ipq2,networkFilename,propertyFilename)


 #   def readProperty(self,filename):
 #       MarabouCore.PropertyParser().parse(filename,self.ipq)



    def tightenBounds(self):
        # Re-tightens bounds on variables from the Input Query computed directly from the file (more accurate)
        self.tightenInputBounds()
        self.tightenOutputBounds()
        self.tighten_fBounds()
        self.tighten_bBounds()

    def tightenInputBounds(self):
        print(self.inputVars)
        for var in self.inputVars.flatten():
            true_lower_bound = self.ipq2.getLowerBound(var)
            true_upper_bound = self.ipq2.getUpperBound(var)
            if self.lowerBounds[var] < true_lower_bound:
                self.setLowerBound(var,true_lower_bound)
                print ('Adjusting lower bound for input variable',var,"to be",true_lower_bound)
            if self.upperBounds[var] > true_upper_bound:
                self.setUpperBound(var,true_upper_bound)
                print ("Adjusting upper bound for input variable",var,"to be",true_upper_bound)

    def tightenOutputBounds(self):
        for var in self.outputVars.flatten():
            true_lower_bound = self.ipq2.getLowerBound(var)
            true_upper_bound = self.ipq2.getUpperBound(var)
            if (not self.lowerBoundExists(var) or self.lowerBounds[var] < true_lower_bound):
                self.setLowerBound(var, true_lower_bound)
                print('Adjusting lower bound for output variable', var, "to be", true_lower_bound)
            if (not self.upperBoundExists(var) or self.upperBounds[var] > true_upper_bound):
                self.setUpperBound(var, true_upper_bound)
                print("Adjusting upper bound for output variable", var, "to be", true_upper_bound)

    def tighten_bBounds(self):
        for var in self.b_variables:
            true_lower_bound = self.ipq2.getLowerBound(var)
            true_upper_bound = self.ipq2.getUpperBound(var)
            if (not self.lowerBoundExists(var) or self.lowerBounds[var] < true_lower_bound):
                self.setLowerBound(var, true_lower_bound)
                #print('Adjusting lower bound for b variable', var, "to be", true_lower_bound)
            if (not self.upperBoundExists(var) or self.upperBounds[var] > true_upper_bound):
                self.setUpperBound(var, true_upper_bound)
                #print("Adjusting upper bound for b variable", var, "to be", true_upper_bound)

    def tighten_fBounds(self):
        for var in self.f_variables:
            true_lower_bound = self.ipq2.getLowerBound(var)
            true_upper_bound = self.ipq2.getUpperBound(var)
            if (not self.lowerBoundExists(var) or self.lowerBounds[var] < true_lower_bound):
                self.setLowerBound(var, true_lower_bound)
                #print('Adjusting lower bound for f variable', var, "to be", true_lower_bound)
            if (not self.upperBoundExists(var) or self.upperBounds[var] > true_upper_bound):
                self.setUpperBound(var, true_upper_bound)
                #print("Adjusting upper bound for f variable", var, "to be", true_upper_bound)

    def testInputBounds(self):
        for var in self.inputVars.flatten():
            print(var, ": between ", self.lowerBounds[var], " and ", self.upperBounds[var])

    def testOutputBounds(self):
        for var in self.outputVars.flatten():
            if self.lowerBoundExists(var) and self.upperBoundExists(var):
                print(var, ": between ", self.lowerBounds[var], " and ", self.upperBounds[var])

    def tightenBounds1(self):
        # Re-tightens bounds on variables from the Input Query computed from the MarabouNetwork object (less accurate)
        self.tightenInputBounds1()
        self.tightenOutputBounds1()
        self.tighten_fBounds1()
        self.tighten_bBounds1()


    def tightenInputBounds1(self):
        for var in self.inputVars.flatten():
             true_lower_bound = self.ipq1.getLowerBound(var)
             true_upper_bound = self.ipq1.getUpperBound(var)
             if self.lowerBounds[var] < true_lower_bound:
                 self.setLowerBound(var,true_lower_bound)
                 print('Adjusting lower bound for input variable', var, "to be", true_lower_bound)
             if self.upperBounds[var] > true_upper_bound:
                 self.setUpperBound(var,true_upper_bound)
                 print("Adjusting upper bound for input variable", var, "to be", true_upper_bound)

    def tightenOutputBounds1(self):
        for var in self.outputVars.flatten():
            true_lower_bound = self.ipq1.getLowerBound(var)
            true_upper_bound = self.ipq1.getUpperBound(var)
            if (not self.lowerBoundExists(var) or self.lowerBounds[var] < true_lower_bound):
                self.setLowerBound(var, true_lower_bound)
                print('Adjusting lower bound for output variable', var, "to be", true_lower_bound)
            if (not self.upperBoundExists(var) or self.upperBounds[var] > true_upper_bound):
                self.setUpperBound(var, true_upper_bound)
                print("Adjusting upper bound for output variable", var, "to be", true_upper_bound)

    def tighten_bBounds1(self):
        for var in self.b_variables:
            true_lower_bound = self.ipq1.getLowerBound(var)
            true_upper_bound = self.ipq1.getUpperBound(var)
            if (not self.lowerBoundExists(var) or self.lowerBounds[var] < true_lower_bound):
                self.setLowerBound(var, true_lower_bound)
                print('Adjusting lower bound for b variable', var, "to be", true_lower_bound)
            if (not self.upperBoundExists(var) or self.upperBounds[var] > true_upper_bound):
                self.setUpperBound(var, true_upper_bound)
                print("Adjusting upper bound for b variable", var, "to be", true_upper_bound)

    def tighten_fBounds1(self):
        for var in self.f_variables:
            true_lower_bound = self.ipq1.getLowerBound(var)
            true_upper_bound = self.ipq1.getUpperBound(var)
            if (not self.lowerBoundExists(var) or self.lowerBounds[var] < true_lower_bound):
                self.setLowerBound(var, true_lower_bound)
                print('Adjusting lower bound for f variable', var, "to be", true_lower_bound)
            if (not self.upperBoundExists(var) or self.upperBounds[var] > true_upper_bound):
                self.setUpperBound(var, true_upper_bound)
                print("Adjusting upper bound for f variable", var, "to be", true_upper_bound)

