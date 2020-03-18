
#from MarabouNetworkNNetIPQ import *
#from MarabouNetworkNNetProperty import *
from MarabouNetworkNNetExtended import *

from Marabou import *

import re
import sys
import parser



import numpy as np



class MarabouNNetMCMH:

    def __init__(self,filename,property_filename):
        self.marabou_nnet = MarabouNetworkNNetExtended(filename=filename,property_filename=property_filename)
        self.ipq = self.marabou_nnet.ipq2
        self.marabou_nnet.tightenBounds()
        self.good_set = []
        self.bad_set = []
        self.layer = -1
        self.layer_bVars = []
        self.layer_fVars = []
        self.layerMinimums = dict()
        self.layerMaximums = dict()
        self.layerfMinimums = dict() #For sanity check
        self.layerfMaximums = dict() #For sanity check


    def setLayer(self,layer):
        assert ((layer>=0) and (layer<self.marabou_nnet.numLayers))
        self.layer = layer
        self.computeLayerVariables(layer)
        self.layerMinimums = {}
        self.layerMaximums = {}

        self.layerfMinimums = {}
        self.layerfMaximums = {}



    def computeLayerVariables(self,layer = -1):
        self.layerVariables = []

        if layer == -1:
            if self.layer > -1:
                layer = self.layer
            else:
                return
        # else:
        #     self.setLayer(layer)
        # # assert ((layer>=0) and (layer<self.marabou_nnet.numLayers)) # Not necessary, happens in setLayer

        for node in range(self.marabou_nnet.layerSizes[layer]):
            self.layer_fVars.append(self.marabou_nnet.nodeTo_f(layer,node))
            self.layer_bVars.append(self.marabou_nnet.nodeTo_b(layer, node))



    def computeHalfLayerVariables(self,layer = -1,f=True):
        self.layerVariables = []

        if layer == -1:
            if self.layer > -1:
                layer = self.layer
            else:
                return
        # else:
        #     self.setLayer(layer)
        # assert ((layer>=0) and (layer<self.marabou_nnet.numLayers)) # Not necessary, happens in setLayer

        for node in self.marabou_nnet.layerSizes[layer]:
            if (f):
                self.layer_fVars.append(self.marabou_nnet.nodeTo_f(layer,node))
            else:
                self.layer_bVars.append(self.marabou_nnet.nodeTo_b(layer, node))

    def createRandomInputs(self):
        inputs = []
        for input_var in self.marabou_nnet.inputVars.flatten():
            assert self.marabou_nnet.upperBoundExists(input_var)
            assert self.marabou_nnet.lowerBoundExists(input_var)
            random_value = np.random.uniform(low=self.marabou_nnet.lowerBounds[input_var],
                                             high=self.marabou_nnet.upperBounds[input_var])
            inputs.append(random_value)
        return inputs

    def clearGoodSet(self):
        self.good_set = []

    def createInitialGoodSet(self,N,adjust_bounds=False,sanity_check=False):
        self.clearGoodSet()
        self.addRandomValuesToGoodSet(N,adjust_bounds,sanity_check)

    def addRandomValuesToGoodSet(self,N,adjust_bounds=False,sanity_check=False):
        assert self.layer >=0
        layer = self.layer

        good_set =[]
        for i in range(N):
            inputs = self.createRandomInputs()

            # Currently running the inputs twice through the networks; perhaps better change this down the road
            if self.badInput(inputs):  # Not normalizing the outputs!
                print('A counter example found! input = ', inputs)

                sys.exit()

            layer_output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=layer, normalize_inputs=False,
                                                                    normalize_outputs=False,activate_output_layer=False)
            good_set.append(layer_output)
            if (adjust_bounds):
                self.adjustLayerBounds(layer_output)

            if (sanity_check):
                layer_output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=layer,
                                                                        normalize_inputs=False,
                                                                        normalize_outputs=False,
                                                                        activate_output_layer=True)
                self.adjustfLayerBounds(layer_output)


        self.good_set += good_set



    def adjustLayerBounds(self,layer_output):
        """
        Adjust lower and upper bounds for the layer
        :param layer_output: list of floats
        :return:
        """
        for i in range(self.marabou_nnet.layerSizes[self.layer]):
            if (not (i in self.layerMinimums)) or (layer_output[i] < self.layerMinimums[i]):
                self.layerMinimums[i] = layer_output[i]
            if (not (i in self.layerMaximums)) or (layer_output[i] > self.layerMaximums[i]):
                self.layerMaximums[i] = layer_output[i]

    def adjustfLayerBounds(self,layer_output):
        """
        Adjust lower and upper bounds for the f-variables of the layer
        Currently used only for sanity check
        :param layer_output: list of floats
        :return:
        """
        for i in range(self.marabou_nnet.layerSizes[self.layer]):
            if (not (i in self.layerfMinimums)) or (layer_output[i] < self.layerfMinimums[i]):
                self.layerfMinimums[i] = layer_output[i]
            if (not (i in self.layerfMaximums)) or (layer_output[i] > self.layerfMaximums[i]):
                self.layerfMaximums[i] = layer_output[i]

    #returns TRUE if variable is within bounds
    #asserts that the variable is legal
    #NOTE: we are assuming that the bounds have been tightened after the property has been incorporated.

    def variableWithinBounds(self,variable,value):
        assert variable >= 0
        assert variable < self.marabou_nnet.numVars

        return  not ((self.marabou_nnet.lowerBoundExists(variable) and \
                value<self.marabou_nnet.lowerBounds[variable]) or \
                (self.marabou_nnet.upperBoundExists(variable) and \
                value>self.marabou_nnet.upperBounds[variable]))


    # If one of the variables in the list  of outputs is out of bounds, returns a list of True and the first such variable
    def outputOutOfBounds(self,output):
        output_variable_index = 0
        for output_variable in self.marabou_nnet.outputVars.flatten():
            if not self.variableWithinBounds(output_variable,output[output_variable_index]):
                    return [True,output_variable]
            output_variable_index+=1
        return [False]


    # Asserts that a legal input is given (all variables are within bounds)
    # returns TRUE if the input is legal (within bounds for input variables) but leads to an illegal output
    def badInput(self,inputs):
        assert len(inputs) == self.marabou_nnet.inputSize
        for input_variable in self.marabou_nnet.inputVars.flatten():
            value = inputs[input_variable]
            assert value >= self.marabou_nnet.lowerBounds[input_variable]
            assert value <= self.marabou_nnet.upperBounds[input_variable]
        output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=-1,normalize_inputs=False,normalize_outputs=False)

        equations_hold = self.marabou_nnet.property.verify_equations_exec(inputs,output)

        return self.outputOutOfBounds(output)[0] or  (not equations_hold)



    def outputVariableToIndex(self,output_variable):
        return output_variable-self.marabou_nnet.numVars+self.marabou_nnet.outputSize


    #Creates a list of outputs of the "extreme" input values
    #Checks whether all these outputs are legal (within bounds and satisfy the equations)
    #Creates a list of "empiric bounds" for the output variables based on the results
    def outputsOfInputExtremes(self):
        outputs = []
        input_size = self.marabou_nnet.inputSize
        output_lower_bounds = dict()
        output_upper_bounds = dict()


        #print 2 ** input_size

        #We don't want to deal with networks that have a large input layer
        assert input_size < 20

        for i in range(2 ** input_size):
            #turning the number i into a bit string of a specific length
            bit_string =  '{:0{size}b}'.format(i,size=input_size)

            #print bit_string #debugging

            inputs = [0 for i in range(input_size)]

            # creating an input: a sequence of lower and upper bounds, determined by the bit string
            for input_var in self.marabou_nnet.inputVars.flatten():
                if bit_string[input_var] == '1':
                    assert self.marabou_nnet.upperBoundExists(input_var)
                    inputs[input_var] = self.marabou_nnet.upperBounds[input_var]
                else:
                    assert self.marabou_nnet.lowerBoundExists(input_var)
                    inputs[input_var] = self.marabou_nnet.lowerBounds[input_var]

            print ("input = ", inputs)

            #Evaluating the network on the input

            output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=-1,normalize_inputs=False,normalize_outputs=False)
            print("output = ", output)
            outputs.append(output)

            if self.outputOutOfBounds(output)[0]: #NOT Normalizing outputs!
                print('A counterexample found! input = ', inputs)

                sys.exit()


            if not self.marabou_nnet.property.verify_equations_exec(inputs,output): #NOT Normalizing outputs!
                print('A counterexample found! input = ', inputs)

                sys.exit()



            # Updating the smallest and the largest ouputs for output variables
            for output_var in self.marabou_nnet.outputVars.flatten():
                output_var_index = self.outputVariableToIndex(output_var)
                if not output_var in output_lower_bounds or output_lower_bounds[output_var]>output[output_var_index]:
                    output_lower_bounds[output_var] = output[output_var_index]
                if not output_var in output_upper_bounds or output_upper_bounds[output_var]<output[output_var_index]:
                    output_upper_bounds[output_var] = output[output_var_index]


        #print len(outputs)
        print ("lower bounds = ", output_lower_bounds)
        print ("upper bounds = ", output_upper_bounds)

        #print(outputs)


    #Creates a list of outputs a given layer for the "extreme" input values
    #Creates a list of "empiric bounds" for the output of the layer based on the results
    def outputsOfInputExtremesForLayer(self, adjust_bounds = True, add_to_goodset = False, sanity_check = False):
        layer_outputs = []
        input_size = self.marabou_nnet.inputSize
        # layer_lower_bounds = dict()
        # layer_upper_bounds = dict()


        #We don't want to deal with networks that have a large input layer
        assert input_size < 20

        for i in range(2 ** input_size):
            #turning the number i into a bit string of a specific length
            bit_string =  '{:0{size}b}'.format(i,size=input_size)

            #print bit_string #debugging

            inputs = [0 for i in range(input_size)]

            # creating an input: a sequence of lower and upper bounds, determined by the bit string
            for input_var in self.marabou_nnet.inputVars.flatten():
                if bit_string[input_var] == '1':
                    assert self.marabou_nnet.upperBoundExists(input_var)
                    inputs[input_var] = self.marabou_nnet.upperBounds[input_var]
                else:
                    assert self.marabou_nnet.lowerBoundExists(input_var)
                    inputs[input_var] = self.marabou_nnet.lowerBounds[input_var]

            # print ("Evaluating layer; input = ", inputs)

            #Evaluating the network up to the given layer on the input
            #By not activating the last layer, we get values for the b variables, which give more information

            output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=self.layer,normalize_inputs=False,normalize_outputs=False,activate_output_layer=False)
            # print("output = ", output)
            layer_outputs.append(output)

            if add_to_goodset:
                self.good_set.append(output)

            if adjust_bounds:
                self.adjustLayerBounds(output)

            if sanity_check:
                output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=self.layer, normalize_inputs=False,
                                                                  normalize_outputs=False, activate_output_layer=True)
                self.adjustfLayerBounds(output)

            # if self.outputOutOfBounds(output)[0]: #NOT Normalizing outputs!
            #     print('A counterexample found! input = ', inputs)
            #
            #     sys.exit()
            #
            #
            # if not self.marabou_nnet.property.verify_equations_exec(inputs,output): #NOT Normalizing outputs!
            #     print('A counterexample found! input = ', inputs)
            #
            #     sys.exit()



            # Updating the smallest and the largest outputs for the layer variables
            # for output_var in self.marabou_nnet.outputVars.flatten():  #CHANGE TO VARS FROM THE GIVEN LAYER!
            #     output_var_index = self.outputVariableToIndex(output_var)
            #     if not output_var in output_lower_bounds or output_lower_bounds[output_var]>output[output_var_index]:
            #         output_lower_bounds[output_var] = output[output_var_index]
            #     if not output_var in output_upper_bounds or output_upper_bounds[output_var]<output[output_var_index]:
            #         output_upper_bounds[output_var] = output[output_var_index]



        #print len(outputs)
        # print ("lower bounds = ", output_lower_bounds)
        # print ("upper bounds = ", output_upper_bounds)

        #print(outputs)


    def createOutputPropertyFileForLayer(self,ouput_property_filename: str):
        """
        Create a property filename for a network in which self.layer is the output layer
        Assumes that the layer is not activated
        Encodes the empiric lower and upper bounds for the b-variables of the layer, which
        are currently stored in the dictionaries self.layerMinimums and self.layerMaximums

        NOTE that self.layer is a hidden layer, so only positive bounds matter and are stored in the property!

        The names of the variables in the property file are going to be yi, where i is the index
        of the variable in the layer

        :param ouput_property_filename: str (the property file to be written into)
        :return:
        """

        with open(ouput_property_filename, 'w') as f2:

            for i in range(self.marabou_nnet.layerSizes[self.layer]):
                if (i in self.layerMinimums) and (self.layerMinimums[i]>0):
                    f2.write("y")
                    f2.write(str(i))
                    f2.write(" >= ")
                    f2.write(str(self.layerMinimums[i]))
                    f2.write("\n")
                if (i in self.layerMaximums):
                    upper_bound = max(self.layerMaximums[i],0.0)
                    f2.write("y")
                    f2.write(str(i))
                    f2.write(" <= ")
                    f2.write(str(upper_bound))
                    f2.write("\n")

    def createInputPropertyFileForLayer(self,input_property_filename: str, sanity_check=False):
        """
        Create a property filename for a network in which self.layer is the input layer
        Assumes that the previous layer has been activated
        Encodes the empiric lower and upper bounds for the f-variables of the layer, which
        are computed from the bounds currently stored in the dictionaries self.layerMinimums and self.layerMaximums

        The names of the variables in the property file are going to be xi, where i is the index
        of the variable in the layer

        :param input_property_filename (str): the property file to be written into
        :param sanity_check (bool): if is True, uses self.layerfMinimums and self.layerfMaximums, for comparison
        :return:
        """

        if not sanity_check:
            layerMinimums = self.layerMinimums
            layerMaximums = self.layerMaximums
        else:
            layerMinimums = self.layerfMinimums
            layerMaximums = self.layerfMaximums


        with open(input_property_filename, 'w') as f2:

            for i in range(self.marabou_nnet.layerSizes[self.layer]):
                if i in layerMinimums:
                    f2.write("x")
                    f2.write(str(i))
                    f2.write(" >= ")
                    lower_bound = max(layerMinimums[i],0.0)
                    f2.write(str(lower_bound))
                    f2.write("\n")
                if i in layerMaximums:
                    f2.write("x")
                    f2.write(str(i))
                    f2.write(" <= ")
                    upper_bound = max(layerMaximums[i], 0.0)
                    f2.write(str(upper_bound))
                    f2.write("\n")





network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_1_9.nnet"
property_filename = "../resources/properties/acas_property_4.txt"
property_filename1 = "../resources/properties/acas_property_1.txt"
#network_filename = "../maraboupy/regress_acas_nnet/ACASXU_run2a_1_7_batch_2000.nnet"


nnet_object = MarabouNNetMCMH(filename=network_filename,property_filename=property_filename)
nnet_object.marabou_nnet.property.compute_executables()

#print(nnet_object.marabou_nnet.property.exec_bounds)
#print(nnet_object.marabou_nnet.property.exec_equations)

# nnet_object.outputsOfInputExtremes()

nnet_object.setLayer(layer=6)

nnet_object.createInitialGoodSet(N=1000,adjust_bounds=True,sanity_check=True)

print(nnet_object.layerMinimums)
print(nnet_object.layerMaximums)


nnet_object.outputsOfInputExtremesForLayer(adjust_bounds=True,add_to_goodset=True,sanity_check=True)
print(nnet_object.layerMinimums)
print(nnet_object.layerMaximums)

output_property_file = "output_property_test1.txt"
input_property_file = "input_property_test1.txt"
input_property_file_sanity = "input_property_test2.txt"

nnet_object.createOutputPropertyFileForLayer(output_property_file)
nnet_object.createInputPropertyFileForLayer(input_property_file)
nnet_object.createInputPropertyFileForLayer(input_property_file_sanity,sanity_check=True)

nnet_object = MarabouNetworkNNetExtended()
print(nnet_object.numLayers)


# solve_query(ipq, filename="", verbose=True, timeout=0, verbosity=2)





