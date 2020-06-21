
'''
/* *******************                                                        */
/*! \file MarabouNetworkNNetExtensions.py
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
 ** Implements several operations on objects of types MarabouNetworkNNet
 ** Main feature currently implemented here is splitting a network of type into two
 **
 ** [[ Add lengthier description here ]]
 **/
'''

import warnings

try:
    from MarabouNetworkNNetQuery import *
except ImportError:
    try:
        from maraboupy.MarabouNetworkNNetQuery import *
    except ImportError:
        warnings.warn('Module MarabouNNetQuery not installed.')

try:
    from MarabouNetworkNNet import *
except ImportError:
    from maraboupy.MarabouNetworkNNet import *
    
import numpy as np


def splitList(list,l):
    return list[:l], list[l:]



def splitNNet(marabou_nnet: MarabouNetworkNNet, layer: int):
    '''
    Takes one MarabouNetworkNNEt object and layer, and returns two MarabouNetworkNNet objects, generated by cutting the
    original network after the given layer.
    Note that the input layer is considered layer 0.
    '''
    if (layer < 0) or (layer > marabou_nnet.numLayers-1):
        warnings.warn("Can not split the network given layer = ", layer)
        return None, None
    weights1, weights2 = splitList(marabou_nnet.weights, layer)
    biases1, biases2 = splitList(marabou_nnet.biases, layer)

    # layerSizes1, layerSizes2 = splitList(marabou_nnet.layerSizes,layer+1)
    # Chose not to implement the split here and compute later; works better.

    new_input_size = marabou_nnet.layerSizes[layer + 1]

    mins1 = marabou_nnet.inputMinimums[:]

    maxs1 = marabou_nnet.inputMaximums[:]

    means1 = marabou_nnet.inputMeans[:]
    ranges1 = marabou_nnet.inputRanges[:]

    '''
    No normalization for the outputs of the first network
    '''
    outputmean1 = 0
    outputrange1 = 1

    '''
    The mins and maxs of the second input layer are taken to be the lower and the upper bounds of b 
    variables corresponding to that layer, respectively
    '''

    # Note: the mins and the maxs for the second network can (and most probably will) contain None values!
    # Should not matter at the moment, since we should never normalize the inputs for the second layer.

    mins2, maxs2 = marabou_nnet.getBoundsForLayer(layer, f=False)

    # maxs2 = [0]*new_input_size  # Change!
    # mins2 = [0]*new_input_size  # Change!

    '''
    No normalization for the new input layer
    '''
    means2 = [0] * (new_input_size)
    ranges2 = [1] * (new_input_size)

    '''
    The mean and the range for the output for the second network are the mean and the range of 
    the original output
    '''
    outputmean2 = marabou_nnet.outputMean
    outputrange2 = marabou_nnet.outputRange

    try:
        marabou_nnet1 = MarabouNetworkNNetQuery(normalize=True)
        marabou_nnet2 = MarabouNetworkNNetQuery(normalize=True)
    except NameError:
        marabou_nnet1 = MarabouNetworkNNet(normalize=True)
        marabou_nnet2 = MarabouNetworkNNet(normalize=True)


    marabou_nnet1.resetNetworkFromParameters(mins1, maxs1, means1, ranges1, outputmean1, outputrange1, weights1, biases1)
    marabou_nnet2.resetNetworkFromParameters(mins2, maxs2, means2, ranges2, outputmean2, outputrange2, weights2, biases2)

    return marabou_nnet1, marabou_nnet2


def createRandomInputsForNetwork(marabou_nnet: MarabouNetworkNNet):

        inputs = []
        for input_var in marabou_nnet.inputVars.flatten():
            assert marabou_nnet.upperBoundExists(input_var)
            assert marabou_nnet.lowerBoundExists(input_var)
            random_value = np.random.uniform(low=marabou_nnet.lowerBounds[input_var],
                                             high=marabou_nnet.upperBounds[input_var])
            inputs.append(random_value)
        return inputs


def computeRandomOutputs(marabou_nnet: MarabouNetworkNNet, N: int):
        output_set =[]
        for i in range(N):
            inputs = createRandomInputsForNetwork(marabou_nnet)


            layer_output = marabou_nnet.evaluateNetwork(inputs, normalize_inputs=False, normalize_outputs=False)
            output_set.append(layer_output)

        return output_set


def computeRandomOutputsToLayer(marabou_nnet: MarabouNetworkNNet,layer: int, N: int):
        output_set =[]
        for i in range(N):
            inputs = createRandomInputsForNetwork(marabou_nnet)


            layer_output = marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=layer, normalize_inputs=False, normalize_outputs=False)
            output_set.append(layer_output)

        return output_set


def test_split_network(nnet_object: MarabouNetworkNNet, nnet_object1: MarabouNetworkNNet, nnet_object2: MarabouNetworkNNet, N = 1000, layer = -1):
    '''
    Runs N random inputs through the first network and subsequently through the two networks smaller (1 and 2) 
    that supposedly were created by splitting the first one, in several different ways.
    Verifies that the results are all the same. 
    
    :param nnet_object: 
    :param nnet_object1: 
    :param nnet_object2: 
    :param N: 
    :param layer:
    :return: 
    '''
    for i in range(N):
            inputs = createRandomInputsForNetwork(nnet_object)

            output1 = nnet_object1.evaluateNetworkToLayer(inputs, last_layer=-1, normalize_inputs=False,
                                                          normalize_outputs=False, activate_output_layer=True)

            # Comparing the output of the first network to the output to layer of the original one
            if layer>=0:
                layer_output = nnet_object.evaluateNetworkToLayer(inputs, last_layer=layer, normalize_inputs=False,
                                                                  normalize_outputs=False, activate_output_layer=True)
                if not (layer_output == output1).all():
                       print("Failed1")
                output2b = nnet_object.evaluateNetworkFromLayer(layer_output,first_layer=layer)


            # The main comparison: comparing running the input through the original network to running it
            # through the first followed by the second.
            true_output = nnet_object.evaluateNetworkToLayer(inputs, last_layer=-1, normalize_inputs=False,
                                                             normalize_outputs=False)
            true_output_b = nnet_object.evaluateNetwork(inputs,normalize_inputs=False,normalize_outputs=False)

            output2 = nnet_object2.evaluateNetworkToLayer(output1,last_layer=-1, normalize_inputs=False,
                                                          normalize_outputs=False)

            if not (true_output == output2).all():
                   print("Failed2")
            if not (true_output_b == output2).all():
                   print("Failed2")


            # Comparing the output of the second network to the output from layer of the original one
            if layer>=0:
                output2b = nnet_object.evaluateNetworkFromLayer(output1,first_layer=layer)
                if not (output2b == true_output).all():
                       print("Failed3")

            #Test evaluateWithoutMarabou from MarabouNetwork.py
            true_output_c = nnet_object.evaluate(np.array([inputs]),useMarabou=False).flatten().tolist()

            if not (true_output_c == output2).all():
                   print("Failed4")
