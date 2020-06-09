

'''
/* *******************                                                        */
/*! \file MarabouNetworkNNetExtended.py
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
 ** This class extends MarabouNetworkNNet class.
 ** Inherits from both MarabouNetworkNNetIPQ and MarabouNetworkNNetProperty
 **
 ** [[ Add lengthier description here ]]
 **/
'''



from MarabouNetworkNNetProperty import *
from  MarabouNetworkNNetIPQ import *

class MarabouNetworkNNetExtended(MarabouNetworkNNetIPQ,MarabouNetworkNNetProperty):
    """
    Class that implements a MarabouNetwork from an NNet file
    Includes extended features: Property and IPQ
    """
    def __init__ (self, filename="", property_filename = "", use_nlr = False, compute_ipq = False):
        """
        Constructs a MarabouNetworkNNetExtended object from an .nnet file.

        Read a property from a property file and stores as a Property object
        Computes the input query and stores as an InputQuery object

        By default computes Input Query from the nnet (and property) files, not from the MarabouNetwork object
        (this results in a more accurate input query)


        Args:
            filename: path to the .nnet file.
            property_filename: path to the property file


        Attributes from MarabouNetworkNNetIPQ:
            ipq1             an Input Query object computed from the network (the MarabouNetwork object)
            ipq2             an Input Query object created from the file (and maybe a property file); more accurate.

        Attributes from MarabouNetworkNNetProperty:
            property         Property object

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
        print('property_filename = ', property_filename)
        super(MarabouNetworkNNetExtended,self).__init__(filename=filename,property_filename=property_filename,use_nlr=use_nlr,compute_ipq=compute_ipq)