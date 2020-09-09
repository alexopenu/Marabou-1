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

#from maraboupy.Marabou import *
from maraboupy import Marabou

from maraboupy.MarabouNetworkNNet import *
from maraboupy.Property import *

#from MarabouNetwork import *
#from MarabouUtils import *
#import MarabouCore


class MarabouNetworkQuery:
    """

    """

    def __init__(self, network_filename='', property_filename='', compute_ipq=True, compute_ipq_directly=False,
                 tighten_bounds = False, verbosity=0):

        self.network_filename = network_filename
        self.property_filename = property_filename
        self.ipq = MarabouCore.InputQuery()
        self.property = Property(property_filename='')

        if network_filename:
            self.setNetworkFilename(network_filename=network_filename, compute_attributes=compute_ipq)
        else:
            self.nnet = MarabouNetworkNNet()

        if property_filename:
            self.setPropertyFilename(property_filename, adjust_ipq=compute_ipq)

        if compute_ipq_directly:
            self.computeIPQDirectly()
            # Note: overrides the existing ipq!

        if tighten_bounds:
            self.tightenBounds(verbosity=verbosity)

    def computeIPQ(self):
        """

        Returns:

        """
        if self.network_filename:
            self.ipq = self.nnet.getMarabouQuery()

    def computeIPQDirectly(self):
        """

        Returns:
        :meta private:
        """
        if self.network_filename:
            MarabouCore.createInputQuery(self.ipq, self.network_filename, self.property_filename, True)

    def setNetworkFilename(self, network_filename: str, compute_attributes=True):
        """

        Args:
            network_filename:
            compute_attributes:

        Returns:

        """
        try:
            self.nnet = Marabou.read_nnet(network_filename, normalize=False)
        except:
            warnings.warn('Something went wrong with reading the network file')
            return
        self.network_filename = network_filename
        if compute_attributes:
            self.computeIPQ()

    def setPropertyFilename(self, property_filename: str, adjust_ipq=True):
        """

        Args:
            property_filename:
            adjust_ipq:

        Returns:

        """
        #try:
        if True:
            self.property = Property(property_filename=property_filename, compute_marabou_lists=False)
        else:
        #except:
            warnings.warn('Something went wrong with reading the property file')
            return

        self.property_filename = property_filename

    def tightenBounds(self, verbosity=0):

        print(self.nnet.getBoundsForLayer(0))
        print([self.ipq.getLowerBound(var) for var in range(self.nnet.layerSizes[0])])
        print([self.ipq.getUpperBound(var) for var in range(self.nnet.layerSizes[0])])

        for var in range(self.nnet.numberOfVariables()):
            try:
                ubd = self.ipq.getUpperBound(var)
            except:
                ubd = sys.float_info.max
                print(var,ubd)
            try:
                lbd = self.ipq.getLowerBound(var)
            except:
                lbd = -sys.float_info.max
                print(var,lbd)
            if True: #not self.nnet.upperBoundExists(var) or ubd < self.nnet.upperBounds[var]:
                self.nnet.setUpperBound(var,ubd)
                if verbosity>2:
                    print('Variable ', var, 'upper bound adjusted to ', ubd)
            if True: #not self.nnet.lowerBoundExists(var) or lbd > self.nnet.lowerBounds[var]:
                self.nnet.setLowerBound(var,lbd)
                if verbosity>2:
                    print('Variable ', var, 'lower bound adjusted to ', lbd)



    def adjustIPQFromProperty(self):
        if not self.property_filename:
            return

        if not self.property.marabou_property_objects_computed:
            self.property.computeMarabouPropertyObjects()

        # TODO: Complete!
