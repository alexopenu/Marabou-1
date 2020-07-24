

'''
/* *******************                                                        */
/*! \file MarabouNetworkQuery.py
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
 ** This class represents a property of a neural network to be verified
 **
 ** [[ Add lengthier description here ]]
 **/
'''

import warnings

import Marabou
from MarabouNetworkNNet import *

from Property import *



from MarabouNetwork import *
from MarabouUtils import *
import MarabouCore

class MarabouNetworkQuery:

    def __init__(self, network_filename = '', property_filename = '', compute_ipq = True):

        self.network_filename = network_filename
        self.property_filename = property_filename
        self.ipq = MarabouCore.InputQuery()

        if network_filename:
            self.setNetworkFilename(network_filename=network_filename, compute_attributes=compute_ipq)
        else:
            self.nnet = MarabouNetworkNNet()

        if property_filename:
            self.setPropertyFilename(property_filename, adjust_ipq=compute_ipq)
        else:
            self.property = Property(property_filename='')

    def computeIPQ(self):
        if self.network_filename:
            self.ipq = self.nnet.getMarabouQuery()


    def setNetworkFilename(self, network_filename: str, compute_attributes = True):
        try:
            self.nnet = Marabou.read_nnet(network_filename)
        except:
            warnings.warn('Something went wrong with reading the network file')
            return
        self.network_filename = network_filename
        if compute_attributes:
            self.computeIPQ()


    def setPropertyFilename(self, property_filename: str, adjust_ipq = True):
        try:
            self.property = Property(property_filename=property_filename)
        except:
            warnings.warn('Something went wrong with reading the network file')
            return

        self.property_filename = property_filename

    def adjustIPQFromProperty(self):
        if not self.property_filename:
            return

        if not self.property.marabou_property_objects_computed:
            self.property.computeMarabouPropertyObjects()

