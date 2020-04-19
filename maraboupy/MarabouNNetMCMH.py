
#from MarabouNetworkNNetIPQ import *
#from MarabouNetworkNNetProperty import *
from MarabouNetworkNNetExtended import *

from Marabou import *
from MarabouNetworkNNetExtentions import *

# import re
import sys
# import parser


import time
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
import scipy.stats as stats
from statsmodels.nonparametric.api import KDEUnivariate
from random import randint



class MarabouNNetMCMH:

    def __init__(self,filename,property_filename,layer=-1):
        self.marabou_nnet = MarabouNetworkNNetExtended(filename=filename,property_filename=property_filename)

        self.network_filename = filename
        self.property_filename = property_filename

        # The input query corresponding to the network+property, computed from the files by Marabou query parse.
        # Storing locally for convenience.
        self.ipq = self.marabou_nnet.ipq2

        # Making sure that all the bounds on the input variables have been computed
        self.marabou_nnet.tightenBounds()
        for input_var in self.marabou_nnet.inputVars.flatten():
            assert self.marabou_nnet.upperBoundExists(input_var)
            assert self.marabou_nnet.lowerBoundExists(input_var)

        self.marabou_nnet.property.compute_executables(recompute=True)

        # List of properties (interpolant candidate) for the chosen layer
        # The properties are recorded as equations/bounds on input variables and stored as strings
        self.interpolant_candidate = []

        # List that contains the interpolant properties recorded as equations/bounds on output variables
        # is supposed to be identical to interpolant_candidate except that the inequalities are flipped
        # (and all occurrences of 'x' is replaced with 'y')
        self.output_interpolant_candidate = []

        self.good_set = []
        self.bad_set = []

        # The hidden layer we will study an invariant for
        # Currently we only work with one hidden layer
        if layer>-1:
            self.setLayer(layer)
        else:
            self.layer = -1

        #Lests of the variables corresponding to the layer
        self.layer_bVars = []
        self.layer_fVars = []

        # Attributes representing minimal and maximal values seen for the variables of the layer
        # The default ones are for the b-variables
        self.layerMinimums = dict()
        self.layerMaximums = dict()
        self.layerfMinimums = dict() #For sanity check
        self.layerfMaximums = dict() #For sanity check

        # we store executable versions of certain properties (that will be evaluated many times) locally
        # for the sake of efficiency
        assert self.marabou_nnet.property.exec_equations_computed
        assert self.marabou_nnet.property.exec_bounds_computed

        self.input_equations = self.marabou_nnet.property.get_exec_input_equations()
        self.output_equations = self.marabou_nnet.property.get_exec_output_equation()
        self.output_bounds = self.marabou_nnet.property.get_exec_output_bounds()



        # The number of inputs that have been considered: used for computation of statistics
        # do we need this? Check.
        self.numberOfInputs = 0

        # Filenames for the new networks created by splitting
        self.network_filename1 = ''
        self.network_filename2 = ''
        self.property_filename1 = ''
        self.property_filename2 = ''

        # various statistics for layer outputs
        self.good_matrix = np.array([])

        self.mean = dict()
        self.epsiloni = dict()
        self.epsiloni_left = dict()
        self.epsiloni_right = dict()

        # the following statistics are currenty not used
        self.median = dict()
        self.sigma = dict()
        self.sigma_left = dict()
        self.sigma_right = dict()
        self.msigma_left = dict()
        self.msigma_right = dict()
        self.range = dict()
        self.maxsigma = 0
        self.maxsigmaleft = 0
        self.maxsigmaright = 0
        self.kde = dict()
        self.kde_eval = dict()
        self.kde1 = dict()
        self.kdedens1 = dict()
        self.cdf = dict()
        self.icdf = dict()
        self.kde2 = dict()
        self.icdf2 = dict()

        # Error tolerance for negating the interpolant
        # Currently not used ; we use epsiloni instead
        self.epsilon = 0.04


    def setLayer(self,layer):
        assert ((layer>=0) and (layer<self.marabou_nnet.numLayers))
        self.layer = layer
        self.computeLayerVariables()

        self.good_set = []
        self.layerMinimums = {}
        self.layerMaximums = {}

        self.layerfMinimums = {}
        self.layerfMaximums = {}

        self.mean = {}
        self.sigma = {}
        self.epsiloni = {}
        self.epsiloni_left = {}
        self.epsiloni_right = {}


    def setEpsilon(self,epsilon: float):
        self.epsilon = epsilon


    def computeLayerVariables(self):
        self.layer_bVars= []
        self.layer_fVars = []


        # if layer == -1:
        #     if self.layer > -1:
        #         layer = self.layer
        #     else:
        #         return
        # else:
        #     self.setLayer(layer)
        # # assert ((layer>=0) and (layer<self.marabou_nnet.numLayers)) # Not necessary, happens in setLayer

        layer = self.layer

        for node in range(self.marabou_nnet.layerSizes[layer]):
            self.layer_fVars.append(self.marabou_nnet.nodeTo_f(layer,node))
            self.layer_bVars.append(self.marabou_nnet.nodeTo_b(layer, node))



    # def computeHalfLayerVariables(self,layer = -1,f=True):
    #     self.layerVariables = []
    #
    #     if layer == -1:
    #         if self.layer > -1:
    #             layer = self.layer
    #         else:
    #             return
    #     # else:
    #     #     self.setLayer(layer)
    #     # assert ((layer>=0) and (layer<self.marabou_nnet.numLayers)) # Not necessary, happens in setLayer
    #
    #     for node in self.marabou_nnet.layerSizes[layer]:
    #         if (f):
    #             self.layer_fVars.append(self.marabou_nnet.nodeTo_f(layer,node))
    #         else:
    #             self.layer_bVars.append(self.marabou_nnet.nodeTo_b(layer, node))

    # NOTE: assumes (for efficiency) that all lower and upper bounds exist
    def createRandomInputs(self):
        input = []
        for input_var in self.marabou_nnet.inputVars.flatten():
            # assert self.marabou_nnet.upperBoundExists(input_var)
            # assert self.marabou_nnet.lowerBoundExists(input_var)
            random_value = np.random.uniform(low=self.marabou_nnet.lowerBounds[input_var],
                                             high=self.marabou_nnet.upperBounds[input_var])
            input.append(random_value)
        return input

    def clearGoodSet(self):
        self.good_set = []

    def createInitialGoodSet(self,N,include_input_extremes=True, adjust_bounds=True,sanity_check=False):
        self.clearGoodSet()
        if include_input_extremes:
            self.outputsOfInputExtremesForLayer()
        self.addRandomValuesToGoodSet(N,adjust_bounds,sanity_check)


    def verifyInputEquations(self,x):
        for eq in self.input_equations:
            if not eval(eq):
                return False
        return True

    def verifyOutputBounds(self,y):
        for eq in self.output_bounds:
            if not eval(eq):
                return False
        return True

    def verifyOutputEquations(self,y):
        for eq in self.output_equations:
            if not eval(eq):
                return False
        return True

    def verifyOutputProperty(self,y):
        return self.verifyOutputBounds(y) and self.verifyOutputEquations(y)


    def addRandomValuesToGoodSet(self,N,adjust_bounds=True, check_bad_inputs = True, sanity_check=False):
        # assert self.layer >=0
        layer = self.layer

        good_set =[]
        for i in range(N):
            inputs = self.createRandomInputs()

            # The input is chosen uniformly at random from within the bounds of input variables
            # Still need to verify that the equations hold; if not, this is not a "legal" input, we discard it
            if not self.verifyInputEquations(inputs):
                continue

            # Evaluating the network at the layer
            layer_output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=layer, normalize_inputs=False,
                                                                    normalize_outputs=False,activate_output_layer=False)
            # we know that the property holds on the inputs
            # Checking whether it also holds on the outputs; if it does , we have a counterexample!
            # Note that we currently assume that there are no constraints on the hidden layer!

            if check_bad_inputs:
                network_output = self.marabou_nnet.evaluateNetworkFromLayer(output,first_layer=layer,
                                                                            normalize_inputs=False,normalize_outputs=False,
                                                                            activate_output_layer=False)
                if self.verifyOutputProperty(network_output):
                    print('A counter example found! Randomly chosen input = ', inputs, 'output = ', network_output)
                    sys.exit(0)

            '''
            # Currently running the inputs twice through the networks; perhaps better change this down the road
            if self.badInput(inputs):  # Not normalizing the outputs!
                print('A counter example found! input = ', inputs)
                output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=-1, normalize_inputs=False,
                                                                  normalize_outputs=False)
                print('output=',output)
                output = self.marabou_nnet.evaluateWithMarabou(np.array([inputs])).flatten()
                print('output=', output)
                sys.exit()
            '''

            good_set.append(layer_output)
            if (adjust_bounds):
                self.adjustLayerBounds(layer_output)

            if (sanity_check):
                layer_output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=layer,
                                                                        normalize_inputs=False,
                                                                        normalize_outputs=False,
                                                                        activate_output_layer=True)
                self.adjustfLayerBounds(layer_output)


        true_N = len(good_set)

        self.good_set += good_set
        self.numberOfInputs += true_N

        if (true_N<N):
            print('Warning in adding random values to good set: equations failed on some of the random inputs, only ',
                  true_N, ' out of ', N, ' inputs were added')




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


    # def inputEquationsHold(self,x):




    # Asserts that a legal input is given (all input variables are within bounds)
    # returns TRUE if the input satisfies all the bounds and equations
    # This method is not used in the current version
    def badInput(self,inputs):
        assert len(inputs) == self.marabou_nnet.inputSize
        for input_variable in self.marabou_nnet.inputVars.flatten():
            value = inputs[input_variable]
            assert value >= self.marabou_nnet.lowerBounds[input_variable]
            assert value <= self.marabou_nnet.upperBounds[input_variable]
        output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=-1,normalize_inputs=False,normalize_outputs=False)

        equations_hold = self.marabou_nnet.property.verify_equations_exec(inputs,output)

        return (not self.outputOutOfBounds(output)[0]) and equations_hold



    def outputVariableToIndex(self,output_variable):
        return output_variable-self.marabou_nnet.numVars+self.marabou_nnet.outputSize





    # Creates a list of outputs for self.layer for the "extreme" input values
    # Creates a list of "empiric bounds" for the output of the layer based on the results
    def outputsOfInputExtremesForLayer(self, adjust_bounds = True, add_to_goodset = True, add_to_statistics = True,
                                       verify_property = True, sanity_check = False):
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
                    # assert self.marabou_nnet.upperBoundExists(input_var)
                    inputs[input_var] = self.marabou_nnet.upperBounds[input_var]
                else:
                    # assert self.marabou_nnet.lowerBoundExists(input_var)
                    inputs[input_var] = self.marabou_nnet.lowerBounds[input_var]

            # print ("Evaluating layer; input = ", inputs)

            #Evaluating the network up to the given layer on the input
            #By not activating the last layer, we get values for the b variables, which give more information

            output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=self.layer,normalize_inputs=False,normalize_outputs=False,activate_output_layer=False)
            # print("output = ", output)
            layer_outputs.append(output)

            if verify_property:
                if self.layer == self.marabou_nnet.numLayers - 1:
                    network_output = output
                else:
                    network_output = self.marabou_nnet.evaluateNetworkFromLayer(output,first_layer=self.layer)

                if self.marabou_nnet.property.verify_io_property(x=inputs,y=network_output):
                    print('A counterexample found! One of the extreme values. Bit string = ', bit_string,
                          '; input = ',inputs, 'output = ', network_output)
                    sys.exit(0)

            if add_to_goodset:
                self.good_set.append(output)

            if adjust_bounds:
                self.adjustLayerBounds(output)

            if sanity_check:
                output = self.marabou_nnet.evaluateNetworkToLayer(inputs, last_layer=self.layer, normalize_inputs=False,
                                                                  normalize_outputs=False, activate_output_layer=True)
                self.adjustfLayerBounds(output)


        if add_to_statistics:
            self.numberOfInputs += 2 ** input_size

        return layer_outputs


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



        #p rint len(outputs)
        # print ("lower bounds = ", output_lower_bounds)
        # print ("upper bounds = ", output_upper_bounds)

        #print(outputs)




    def create_new_xPropertyFile(self,property_filename1):

        assert self.property_filename1

        print('Warning: replacing property_filename1, from ', self.property_filename1, " to ", property_filename1)

        x_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'x']

        try:
            with open(property_filename1,'w') as f2:
                for l in x_properties:
                    f2.write(l)
        except:
            print('Something went wrong with creating the initial property files or writing to them')
            sys.exit(1)

        self.property_filename1 = property_filename1


    def create_new_yPropertyFile(self, property_filename2):

        assert self.property_filename2

        print('Warning: replacing property_filename2, from ', self.property_filename2, " to ", property_filename2)

        y_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'y']

        try:
            with open(property_filename2, 'w') as f2:
                for l in y_properties:
                    f2.write(l)
        except:
            print('Something went wrong with creating the initial property files or writing to them')
            sys.exit(1)

        self.property_filename2 = property_filename2


    # I think this method is redundant
    def createPropertyFiles(self,property_filename1,property_filename2):
        # if self.marabou_nnet.property.bounds['ws'] or self.marabou_nnet.property.equations['m'] or
        #     self.marabou_nnet.property.equations['ws']:
        #     print('Mixed equations and bounds on hidden variables currently not supported')
        #     sys.exit(1)

        x_properties = []
        y_properties = []

        for p in self.marabou_nnet.property.properties_list:
            if p['type2'] == 'x':
                x_properties.append(p['line'])
            elif p['type2'] == 'y':
                y_properties.append(p['line'])
            else:
                print('Only pure input and output properties are currently supported')
                sys.exit(1)

        # input_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'x']
        # output_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'y']

        try:
            with open(property_filename1,'w') as f2:
                for l in x_properties:
                    f2.write(l)
            with open(property_filename2,'w') as f2:
                for p in y_properties:
                    f2.write(l)
        except:
            print('Something went wrong with creating the initial property files or writing to them')
            sys.exit(1)

        self.property_filename1 = property_filename1
        self.property_filename2 = property_filename2


    def createInitialPropertyFiles(self,property_filename1,property_filename2):
        '''
        Creates property files for the two networks based on the original property
        properties involving x only are copied to property_filename1
        properties involving y only are copied to property_filename2
        the new filenames are stored in self.property_filename1 and self.property_filename2
        if self.property_filename1 or self.property_filename2 are not empty, throws an exception
        if a property that involves 'ws' (a hidden variable) or a 'mixed' property (i.e., involving both
            x and y) is found, throws an exception

        :param property_filename1: str
        :param property_filename2: str
        :return:
        '''
        assert not self.property_filename1
        assert not self.property_filename2

        x_properties = []
        y_properties = []

        for p in self.marabou_nnet.property.properties_list:
            if p['type2'] == 'x':
                x_properties.append(p['line'])
            elif p['type2'] == 'y':
                y_properties.append(p['line'])
            else:
                print('Only pure input and output properties are currently supported')
                sys.exit(1)

        try:
            with open(property_filename1,'w') as f2:
                for l in x_properties:
                    f2.write(l)
            with open(property_filename2,'w') as f2:
                for p in y_properties:
                    f2.write(l)
        except:
            print('Something went wrong with creating the initial property files or writing to them')
            sys.exit(1)

        # self.create_xPropertyFile(property_filename1)
        # self.create_yPropertyFile(property_filename2)

        self.property_filename1 = property_filename1
        self.property_filename2 = property_filename2


    def computeInitialInterpolantCandidateForLayer(self):
        '''
        computes the initial interpolant candidate, based on the empiric bounds for the self.layer variables
            and on the epsilons computed using statistical analysis of the values of these variables
        the candidate itself is stored in self.interpolant_candidate (list of strings)
            each string is of the form "xi <= ??" or "xi >= ??"
            it is going to be used as the input property for the network whose input layer is self.layer
        the "dual" candidate - with x replaced with y and the inequalities reversed in stored in
            self.output_interpolant_candidate
        :return:
        '''
        if not sanity_check:
            layerMinimums = self.layerMinimums
            layerMaximums = self.layerMaximums
        else:
            layerMinimums = self.layerfMinimums
            layerMaximums = self.layerfMaximums

        interpolant_list = []
        output_interpolant_list = []

        for i in range(self.marabou_nnet.layerSizes[self.layer]):
            if i in layerMinimums:
                individual_property = []
                individual_property.append('x')
                individual_property.append(str(i))
                individual_property.append(" >= ")
                if i in self.epsiloni_left:
                    epsilon = self.epsiloni_left[i]
                else:
                    epsilon = self.epsilon
                lower_bound = max(layerMinimums[i]-epsilon,0.0)
                individual_property.append(str(lower_bound))
                individual_property.append("\n")

                individual_property_string = ''.join(individual_property)

                interpolant_list.append(individual_property_string)

                output_interpolant_list.append(individual_property_string.replace('x','y').replace('>','<'))
            if i in layerMaximums:
                individual_property = []
                individual_property.append("x")
                individual_property.append(str(i))
                individual_property.append(" <= ")
                if i in self.epsiloni_right:
                    epsilon = self.epsiloni_right[i]
                else:
                    epsilon = self.epsilon
                upper_bound = max(layerMaximums[i] + epsilon,0.0)
                # if layerMaximums[i]<0:
                #     upper_bound = 0.0
                # else:
                #     upper_bound = layerMaximums[i]+self.epsiloni_right[i]
                individual_property.append(str(upper_bound))
                individual_property.append("\n")

                individual_property_string = ''.join(individual_property)

                interpolant_list.append(individual_property_string)

                output_interpolant_list.append(individual_property_string.replace('x','y').replace('<','>'))

        self.interpolant_candidate = interpolant_list
        self.output_interpolant_candidate = output_interpolant_list


    def addLayerPropertiesTo_yPropertyFile(self):
        '''
        adds self.interpolant_candidate to self.property_filename2
            whcih is the property file for the network whose input layer is self.layer
        :return:
        '''
        assert self.property_filename2

        try:
            with open(self.property_filename2,'a') as f2:
                for p in self.interpolant_candidate:
                    f2.write(p)
        except:
            print('Something went wrong with writing to property_file2')
            sys.exit(1)

    def addLayerPropertyByIndexTo_xPropertyFile(self,index=0):
        '''
        adds one property (string) from self.output_interpolant_candidate to self.property_filename1
            which is the property file for the network whose output layer is self.layer
        note that what needs to be verified for this network is the disjunction of self.output_interpolant_candidate
            (hence it makes sense to add one at a time)
        :return:
        '''

        assert self.property_filename1

        try:
            with open(self.property_filename1, 'a') as f2:
                p = self.output_interpolant_candidate[index]
                f2.write(p)
        except:
            print('Something went wrong with writing to property_file1')
            sys.exit(1)



    # This method is from an older version, and is now redundant
    def createRandomOutputPropertyFileForLayer(self,output_property_filename: str):
        """
        Create a property filename for a network in which self.layer is the output layer
        Assumes that the layer is not activated
        Encodes the empiric lower and upper bounds for the b-variables of the layer, which
        are currently stored in the dictionaries self.layerMinimums and self.layerMaximums

        NOTE that self.layer is a hidden layer, so only positive bounds matter and are stored in the property!

        The names of the variables in the property file are going to be yi, where i is the index
        of the variable in the layer

        Chooses a disjunct at random (as long as it makes for an "interesting" property)
        and writes just that one into the property file

        :param ouput_property_filename: str (the property file to be written into)
        :return:
        """
        epsilon = self.epsilon

        try:
            with open(output_property_filename, 'w') as f2:

                while(True):
                    i = randint(0,self.marabou_nnet.layerSizes[self.layer])
                    boundary = randint(0,1)


                    # I believe it is correct now

                    if boundary:
                        lower_bound = self.layerMinimums[i] - epsilon
                        if (i in self.layerMinimums) and (lower_bound>0):
                            f2.write("y")
                            f2.write(str(i))
                            f2.write(" <= ")  # NEGATING the property!
                            f2.write(str(lower_bound))
                            f2.write("\n")
                            break;
                    else:
                        if (i in self.layerMaximums):
                            if self.layerMaximums[i]<0:
                                upper_bound = 0.0
                            else:
                                upper_bound = self.layerMaximums[i]+epsilon
                            f2.write("y")
                            f2.write(str(i))
                            f2.write(" >= ")  # NEGATING the property!
                            f2.write(str(upper_bound))
                            f2.write("\n")
                            break;

                with open(self.property_filename, 'r') as f:
                    line = f.readline()
                    while (line):
                        if ('w' in line):
                            print("w_i in the property file, not supported")
                            sys.exit(1)
                        if 'x' in line:
                            if ('y' in line):
                                print("Both x and y in the same equation in the property file, exiting")
                                sys.exit(1)
                            else:
                                f2.write(line)
                        line = f.readline()

        except:
            print("Something went wrong with writing to the output property file",output_property_filename)
            sys.exit(1)

    # This method is from an older version, and is now redundant
    def createSingleOutputPropertyFileForLayer(self,output_property_filename: str, i: int, lb: bool):
        """
        Create a property filename for a network in which self.layer is the output layer
        Assumes that the layer is not activated
        Encodes the empiric lower and upper bounds for the b-variables of the layer, which
        are currently stored in the dictionaries self.layerMinimums and self.layerMaximums

        NOTE that self.layer is a hidden layer, so only positive bounds matter

        The names of the variables in the property file are going to be yi, where i is the index
        of the variable in the layer


        Creates a property file that encodes one disjunct, corresponding to:
        i (int): the index of the variable
        lb (bool): True means lower bound, False means upper bound


        :param ouput_property_filename: str (the property file to be written into)

        :return (bool): True if there is an interesting property to prove, False otherwise
        """
        epsilon = self.epsilon

        interesting_property = False

        try:
            with open(output_property_filename, 'w') as f2:

                if lb:  # Lower bound
                    lower_bound = self.layerMinimums[i] - epsilon
                    if (i in self.layerMinimums) and (lower_bound>0):
                        f2.write("y")
                        f2.write(str(i))
                        f2.write(" <= ")  # NEGATING the property!
                        f2.write(str(lower_bound))
                        f2.write("\n")

                        interesting_property = True

                else:  # Upper bound
                    if (i in self.layerMaximums):
                        if self.layerMaximums[i]<0:
                            upper_bound = 0.0
                        else:
                            upper_bound = self.layerMaximums[i]+epsilon
                        f2.write("y")
                        f2.write(str(i))
                        f2.write(" >= ")  # NEGATING the property!
                        f2.write(str(upper_bound))
                        f2.write("\n")

                        interesting_property = True

                if interesting_property:
                    ''' 
                    Copying the property for the input variables from the original property file
                    '''
                    try:
                        with open(self.property_filename, 'r') as f:
                            line = f.readline()
                            while (line):
                                if ('w' in line):
                                    print("w_i in the property file, not supported")
                                    sys.exit(1)
                                if 'x' in line:
                                    if ('y' in line):
                                        print("Both x and y in the same equation in the property file, exiting")
                                        sys.exit(1)
                                    else:
                                        f2.write(line)
                                line = f.readline()
                    except:
                        print("Something went wrong with copying from the property file to the output property file",
                              output_property_filename)
                        sys.exit(1)


        except:
            print("Something went wrong with writing to the output property file",output_property_filename)
            sys.exit(1)

        return(interesting_property)

    # This method is from an older version, and is now redundant
    def createInputPropertyFileForLayer(self,input_property_filename: str, sanity_check=False):
        """
        Create a property filename for a network in which self.layer is the input layer
        Assumes that the previous layer has been activated
        Encodes the empiric lower and upper bounds for the f-variables of the layer, which
        are computed from the bounds currently stored in the dictionaries self.layerMinimums and self.layerMaximums

        If the upper bound if negative, we change it to 0
        Same for lower bound

        If the upper bound is positive, we add epsilon to it
        If the lower bound is positive, subtract epsilon

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

        try:
            with open(input_property_filename, 'w') as f2:

                for i in range(self.marabou_nnet.layerSizes[self.layer]):
                    if i in layerMinimums:
                        f2.write("x")
                        f2.write(str(i))
                        f2.write(" >= ")
                        if i in self.epsiloni_left:
                            epsilon = self.epsiloni_left[i]
                        else:
                            epsilon = self.epsilon
                        lower_bound = max(layerMinimums[i]-epsilon,0.0)
                        f2.write(str(lower_bound))
                        f2.write("\n")
                    if i in layerMaximums:
                        f2.write("x")
                        f2.write(str(i))
                        f2.write(" <= ")
                        if i in self.epsiloni_right:
                            epsilon = self.epsiloni_right[i]
                        else:
                            epsilon = self.epsilon
                        upper_bound = max(layerMaximums[i] + epsilon,0.0)
                        # if layerMaximums[i]<0:
                        #     upper_bound = 0.0
                        # else:
                        #     upper_bound = layerMaximums[i]+self.epsiloni_right[i]
                        f2.write(str(upper_bound))
                        f2.write("\n")
        except:
            print("Something went wrong with writing to property file2",input_property_filename)
            sys.exit(1)


        self.property_filename2 = input_property_file

    # This method is from an older version, and is now redundant
    def createPropertyFilesForLayer(self,property_filename1: str, property_filename2: str, sanity_check=False):
        '''
        THIS IS WRONG, uses the wrong version of createOuputProperty


        :param property_filename1:
        :param property_filename2:
        :param sanity_check:
        :return:
        '''

        assert self.property_filename
        assert property_filename1
        assert property_filename2

        try:
            self.createInputPropertyFileForLayer(property_filename2,sanity_check)
            self.createOutputPropertyFileForLayer(property_filename1)

            # Appending the input and the output properties from the original property file to the
            # input and the output property files, respectively

            with open(self.property_filename, 'r') as f:
                with open(property_filename1, 'a') as f1:
                    with open(property_filename2, 'a') as f2:
                        line = f.readline()
                        while (line):
                            if 'x' in line:
                                if 'y' in line:
                                    print("Both x and y in the same equation in the property file, exiting")
                                    sys.exit(1)
                                else:
                                    f1.write(line)
                            else:
                                if 'y' in line:
                                    f2.write(line)
                                else:
                                    print("Found an equation in the property file for non input/output variables, exiting")
                                    sys.exit(1)
                            line = f.readline()
                    #
                    # for i in range(self.marabou_nnet.layerSizes[self.layer]):
                    #     if i in layerMinimums:
                    #         f2.write("x")
                    #         f2.write(str(i))
                    #         f2.write(" >= ")
                    #         lower_bound = max(layerMinimums[i], 0.0)
                    #         f2.write(str(lower_bound))
                    #         f2.write("\n")
                    #     if i in layerMaximums:
                    #         f2.write("x")
                    #         f2.write(str(i))
                    #         f2.write(" <= ")
                    #         upper_bound = max(layerMaximums[i], 0.0)
                    #         f2.write(str(upper_bound))
                    #          f2.write("\n")



        except:
            print("Something went wrong with writing to one of the property files")
            sys.exit(1)


        self.property_filename1 = property_filename1
        self.property_filename2 = property_filename2



    def split_network(self, network_filename1: str, network_filename2: str):
        '''
        Splits the network into two
        The split is done after after self.layer
        i.e., self.layer is the last layer of the first network
        Writes both networks into files in nnet format

        :param network_filename1 (str): file to write the first network to (in nnet format)
        :param network_filename2 (str): file to write the second network to (in nnet format)
        :return:
        '''

        assert self.layer >=0

        assert network_filename1
        assert network_filename2

        try:
            nnet_object1, nnet_object2 = splitNNet(marabou_nnet=self.marabou_nnet, layer=self.layer)

            nnet_object1.writeNNet(network_filename1)
            nnet_object2.writeNNet(network_filename2)

        except:
            print("Something went wrong with spltting the network and writing the output networks to files.")

        self.network_filename1 = network_filename1
        self.network_filename2 = network_filename2


    def computeEpsilonsUniformly(self):
        '''
        computes epsilons (to be added to or subtracted from empiric upper and lower bounds for the hidden layer
            variables) based on a very simple statistical heuristics, essentially assuming that the distibution
            on the values of each variable is close to uniform. This is not the case, but the computation is
            efficient, and seems to give "conservative" bounds, which should be a good starting point.
        :return:
        '''
        self.good_matrix = np.array(self.good_set)

        for var in range(self.marabou_nnet.layerSizes[self.layer]):
            self.computeEpsilonsForVariable(var)





    def computeEpsilonsForVariable(self,var):

        outputs = self.good_matrix[:, var]

        sample_size = len(outputs)
        mean = np.mean(outputs)
        min = np.min(outputs)
        max = np.max(outputs)

        self.mean[var] = mean

        self.range[var] = max-min
        self.epsiloni[var] = self.range[var]/sample_size

        small_outputs = [output for output in outputs if (output <= mean)]
        small_range = mean-min
        big_outputs = [output for output in outputs if (output > mean)]
        big_range = max-mean



        self.epsiloni_left[var] = small_range/len(small_outputs)
        self.epsiloni_right[var] = big_range/len(big_outputs)





    def computeStatistics(self):
        self.good_matrix = np.array(self.good_set)

        for var in range(self.marabou_nnet.layerSizes[self.layer]):
            self.estimateStatisticsForVariable(var)

        self.maxsigma = max([self.sigma[i] for i in self.sigma])
        self.maxsigmaleft = max([self.sigma_left[i] for i in self.sigma_left])
        self.maxsigmaright = max([self.sigma_right[i] for i in self.sigma_right])


    def estimateStatisticsForVariable(self,var=0):
        # outputs = sorted([output[var] for output in self.good_set])

        outputs = sorted(self.good_matrix[:,var])

        sample_size = len(outputs)
        mean = np.mean(outputs)

        self.mean[var] = mean

        self.sigma[var] = np.sqrt(sum((outputs - mean) ** 2) / (sample_size - 1))

        small_outputs = [output for output in outputs if (output <= mean)]
        big_outputs = [output for output in outputs if (output > mean)]
        # big_outputs = [outputs[i] for i in self.mean if (outputs[i] > self.mean[i])]
        self.sigma_left[var] = np.sqrt(sum((small_outputs - mean) ** 2) / (len(small_outputs) - 1))
        self.sigma_right[var] = np.sqrt(sum((big_outputs - mean) ** 2) / (len(big_outputs) - 1))

        self.median[var] = outputs[round(sample_size/2)]

        median = self.median[var]

        # a stupid way to compute the sigmas in a sorted array, better rewrite
        small_outputs = [output for output in outputs if (output <= median)]
        big_outputs = [output for output in outputs if (output > median)]
        self.msigma_left[var] = np.sqrt(sum((small_outputs - median) ** 2) / (len(small_outputs) - 1))
        self.msigma_right[var] = np.sqrt(sum((big_outputs - median) ** 2) / (len(big_outputs) - 1))

        self.range[var] = outputs[-1]-outputs[0]
        self.epsiloni[var] = self.range[var]*1/sample_size
        self.epsiloni_left[var] = (outputs[round(sample_size/2)]-outputs[0])*1/len(small_outputs)
        self.epsiloni_right[var] = (-outputs[round(sample_size/2)+1]+outputs[-1])*1/len(big_outputs)

        x_grid = np.linspace(self.layerMinimums[var] - 0.1, self.layerMaximums[var] + 0.1, 1000)

        kde = stats.gaussian_kde(outputs, bw_method='scott')


        self.kde[var] = kde



        self.kde_eval[var] = kde.evaluate(x_grid)

        print("var = ", var, "\n", len(self.kde_eval[var]), "kde_eval: ", self.kde_eval[var][:10])

        kde1 = KDEUnivariate(outputs)

        kde1.fit(kernel='gau', bw='scott', gridsize=1000)

        # self.cdf[var] = kde1.cdf
        self.icdf[var] = kde1.icdf

        self.kde1[var] = kde1
        self.kdedens1[var] = kde1.evaluate(x_grid)

        print(len(self.kdedens1[var]),"kde1_eval: ", self.kdedens1[var][:10])

        kde2 = KDEUnivariate(self.kdedens1[var])

        kde2.fit(kernel='gau', bw='scott', gridsize=1000)
        self.kde2[var] = kde2

        kde2_eval = kde2.evaluate(x_grid)

        print(len(kde2_eval), "kde2_eval: ", kde2_eval[:10])

        self.icdf2[var] = kde2.icdf



    def graphGoodSetDist(self,i=0):
        sns.distplot([output[i] for output in self.good_set],label='Variable'+str(i))
        if i in self.mean:
            x = np.linspace(self.mean[i] - 2 * self.sigma[i], self.mean[i] + 2 * self.sigma[i], 100)
            plt.plot(x, stats.norm.pdf(x, self.mean[i], self.sigma[i]),label = 'Gaussian approximation')
            x_grid = np.linspace(self.layerMinimums[i] - 0.1, self.layerMaximums[i] + 0.1, 1024)
            plt.plot(x_grid, self.kde[i].evaluate(x_grid), color='red', label = 'Gaussian kde')
            plt.plot(x_grid, self.kde1[i].evaluate(x_grid), color='purple', label='Second Gaussian kde')
            plt.plot(x_grid, self.kde2[i].evaluate(x_grid), color='magenta', label='Third Gaussian kde')
            # plt.plot(x_grid, self.cdf[i], color='orange', label='Cumulative Gaussian kde')
            # plt.plot(self.cdf[i], x_grid, color='orange', label='Cumulative Gaussian kde')
            zeroone_grid = np.linspace(0,1,1024)
            plt.plot(self.icdf[i], zeroone_grid, color='green', label='Cumulative Gaussian kde')
            plt.plot(self.icdf2[i], zeroone_grid, color='yellow', label='Second Cumulative Gaussian kde')
        plt.legend()
        plt.show()



    '''
    # NOT USED, and currently wrong (checks the negation of the property on the outputs)
    # See the method outputsOfInputExtremesForLayer, implemented correctly

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


            print("input = ", inputs)

            # Evaluating the network on the input

            output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=-1,normalize_inputs=False,normalize_outputs=False)
            print("output = ", output)
            outputs.append(output)

            # if self.outputOutOfBounds(output)[0]: #NOT Normalizing outputs!
            #     print('A counterexample found! input = ', inputs)
            # 
            #     sys.exit()
            # 
            # 
            # if not self.marabou_nnet.property.verify_equations_exec(inputs,output): #NOT Normalizing outputs!
            #     print('A counterexample found! One of the extreme values. Vector = ',bit_string, '; input = ', inputs)
            # 
            #     sys.exit()
            # 
            # 
            # 
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

'''

    # def createOutputPropertyFileForLayer(self,output_property_filename: str):
    #     """
    #     OK THIS IS TOTALLY WRONG! FIRST, the output property needs to be negated. Second, it is a disjuncion!
    #
    #
    #     Create a property filename for a network in which self.layer is the output layer
    #     Assumes that the layer is not activated
    #     Encodes the empiric lower and upper bounds for the b-variables of the layer, which
    #     are currently stored in the dictionaries self.layerMinimums and self.layerMaximums
    #
    #     NOTE that self.layer is a hidden layer, so only positive bounds matter and are stored in the property!
    #
    #     The names of the variables in the property file are going to be yi, where i is the index
    #     of the variable in the layer
    #
    #     :param ouput_property_filename: str (the property file to be written into)
    #     :return:
    #     """
    #
    #     try:
    #         with open(output_property_filename, 'w') as f2:
    #
    #             for i in range(self.marabou_nnet.layerSizes[self.layer]):
    #                 if (i in self.layerMinimums) and (self.layerMinimums[i]>0):
    #                     f2.write("y")
    #                     f2.write(str(i))
    #                     f2.write(" >= ")
    #                     f2.write(str(self.layerMinimums[i]))
    #                     f2.write("\n")
    #                 if (i in self.layerMaximums):
    #                     upper_bound = max(self.layerMaximums[i],0.0)
    #                     f2.write("y")
    #                     f2.write(str(i))
    #                     f2.write(" <= ")
    #                     f2.write(str(upper_bound))
    #                     f2.write("\n")
    #     except:
    #         print("Something went wrong with writing to the output property file",output_property_filename)
    #         sys.exit(1)


start_time = time.time()

network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet"
property_filename = "../resources/properties/acas_property_4.txt"
property_filename1 = "../resources/properties/acas_property_1.txt"


#network_filename = "../maraboupy/regress_acas_nnet/ACASXU_run2a_1_7_batch_2000.nnet"


mcmh_object = MarabouNNetMCMH(filename=network_filename, property_filename=property_filename)
mcmh_object.marabou_nnet.property.compute_executables()

# solve_query(mcmh_object.marabou_nnet.ipq2,property_filename)

#print(nnet_object.marabou_nnet.property.exec_bounds)
#print(nnet_object.marabou_nnet.property.exec_equations)

# nnet_object.outputsOfInputExtremes()

mcmh_object.setLayer(layer=5)

mcmh_object.createInitialGoodSet(N=1000, adjust_bounds=True, sanity_check=False)

print(mcmh_object.layerMinimums)
print(mcmh_object.layerMaximums)


mcmh_object.outputsOfInputExtremesForLayer(adjust_bounds=True, add_to_goodset=True, sanity_check=False)
print(mcmh_object.layerMinimums)
print(mcmh_object.layerMaximums)

output_property_file = "output_property_test1.txt"
input_property_file = "input_property_test1.txt"
input_property_file_sanity = "input_property_test2.txt"
output_property_file1 = "output_property_test2.txt"

# mcmh_object.createOutputPropertyFileForLayer(output_property_file)

mcmh_object.createInputPropertyFileForLayer(input_property_file)

# mcmh_object.createInputPropertyFileForLayer(input_property_file_sanity, sanity_check=True)
# mcmh_object.createRandomOutputPropertyFileForLayer(output_property_file1)



# mcmh_object.createPropertyFilesForLayer(output_property_file,input_property_file)


# sys.exit(0)

# nnet_object = MarabouNetworkNNetExtended()
# print(nnet_object.numLayers)


# nnet_object1, nnet_object2 = splitNNet(marabou_nnet=mcmh_object.marabou_nnet, layer=layer)


output_filename = "test/ACASXU_experimental_v2a_1_9_output.nnet"
output_filename1 = "test/ACASXU_experimental_v2a_1_9_output1.nnet"
output_filename2 = "test/ACASXU_experimental_v2a_1_9_output2.nnet"


# nnet_object1.writeNNet(output_filename1)
# nnet_object2.writeNNet(output_filename2)


mcmh_object.split_network(output_filename1,output_filename2)

# Testing the random output property file!
# nnet_object1 = MarabouNetworkNNetExtended(output_filename1,output_property_file1)

nnet_object2 = MarabouNetworkNNetExtended(output_filename2,input_property_file)



# Counting wrong answers for the different disjuncts
num_sats = 0
sats = []


# Going over all the disjuncts, one by one
# for var in range(mcmh_object.marabou_nnet.layerSizes[mcmh_object.layer]):
#     for lb in [True,False]:
#         if (mcmh_object.createSingleOutputPropertyFileForLayer(output_property_file1,var,lb)):
#             '''
#             The disjunct leads to a property that needs to be verified
#             '''
#             nnet_object1 = MarabouNetworkNNetExtended(output_filename1, output_property_file1)
#             solution = solve_query(nnet_object1.ipq2, verbosity=0)[0]
#             if solution:  #  SAT; the dict is not empty!
#                 num_sats+=1
#                 string = "lower bound" if lb else "upper bound"
#                 sats.append((var,lb,string))
#




print("Number of SATs: ", num_sats)
print(sats)

# for (var,lb,string) in sats:

# solve_query(nnet_object2.ipq2,verbosity=0)

time1 = time.time()

print("Time taken: ",time.time()-start_time)

# test_split_network(mcmh_object.marabou_nnet,nnet_object1,nnet_object2)
#

mcmh_object.computeStatistics()

time2 = time.time()
print("Time statistics took", time2-time1)
#
# print(mcmh_object.maxsigma,mcmh_object.maxsigmaleft,mcmh_object.maxsigmaright)
# print(mcmh_object.sigma_left)
# print(mcmh_object.sigma_right)
#
# estimated_bounds = dict()
# estimated_lbounds = dict()
# estimated_ubounds = dict()
# for i in range(mcmh_object.marabou_nnet.layerSizes[mcmh_object.layer]):
#     estimated_bounds[i] = (mcmh_object.mean[i]-3*mcmh_object.sigma_left[i],mcmh_object.mean[i]+3*mcmh_object.sigma_right[i])
#     estimated_lbounds[i] = mcmh_object.mean[i]-4*mcmh_object.sigma_left[i]
#     estimated_ubounds[i] = mcmh_object.mean[i]+4*mcmh_object.sigma_right[i]
#
# print("estimated lower bounds: \n ", estimated_lbounds)
# print(mcmh_object.layerMinimums)
# print("estimated upper bounds: \n", estimated_ubounds)
# print(mcmh_object.layerMaximums)



# mcmh_object.graphGoodSetDist(0)
# mcmh_object.graphGoodSetDist(32)
# mcmh_object.graphGoodSetDist(33)
# mcmh_object.graphGoodSetDist(31)


# def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
#     """Kernel Density Estimation with Scipy"""
#     # Note that scipy weights its bandwidth by the covariance of the
#     # input data.  To make the results comparable to the other methods,
#     # we divide the bandwidth by the sample standard deviation here.
#     kde = stats.gaussian_kde(x, bw_method='scott', **kwargs)
#     return kde.evaluate(x_grid)

x_grid = np.linspace(-4.5, 3.5, 1000)

good_set_array = np.array(mcmh_object.good_set)


print(good_set_array[: , 17])

# kde_scipy(good_set_array[: , 17], x_grid)






# for i in range(mcmh_object.marabou_nnet.layerSizes[mcmh_object.layer]):
#     print("variable: ", i)
#     print("mean = ",mcmh_object.mean[i])
#     print("median = ",mcmh_object.median[i])
#     print("sigma = ",mcmh_object.sigma[i])
#     print("sigma left = ",mcmh_object.sigma_left[i])
#     print("sigma right = ",mcmh_object.sigma_right[i])
#
#     print("sigma m left= ",mcmh_object.msigma_left[i])
#     print("sigma m right = ",mcmh_object.msigma_right[i])
#
#     print("sigma = ",mcmh_object.sigma[i])
#
#     print("Observed bounds: ", mcmh_object.layerMinimums[i], mcmh_object.layerMaximums[i])
#
#     print("Bounds based on 3 sigma left and right:",
#           mcmh_object.mean[i]-3*mcmh_object.sigma_left[i], mcmh_object.mean[i]+3*mcmh_object.sigma_right[i])
#
#     print("Bounds based on 3 m sigma left and right:",
#           mcmh_object.median[i]-3*mcmh_object.msigma_left[i], mcmh_object.median[i]+3*mcmh_object.msigma_right[i])
#
#     print("Bounds based on 4 sigma left and right:",
#           mcmh_object.mean[i]-3.5*mcmh_object.sigma_left[i], mcmh_object.mean[i]+3.5*mcmh_object.sigma_right[i])
#
#     print("Bounds based on 4 m sigma left and right:",
#           mcmh_object.median[i]-3.5*mcmh_object.msigma_left[i], mcmh_object.median[i]+3.5*mcmh_object.msigma_right[i])
#
#     print("Bounds computed with epsilon left and right: ",
#           mcmh_object.layerMinimums[i] - mcmh_object.epsiloni_left[i], mcmh_object.layerMaximums[i] + mcmh_object.epsiloni_right[i])
#
#     print("range: ", mcmh_object.range[i])
#
#     print("Epsilons: ", mcmh_object.epsiloni[i],mcmh_object.epsiloni_left[i],mcmh_object.epsiloni_right[i])
#
#     print("Quantiles: ")
#
#     epsilon = 0.0001
#     cdf_sample = len(mcmh_object.icdf[i])
#
#     print("size of cdf sammple: ", cdf_sample)
#
#     print(mcmh_object.icdf[i])
#
#     print("99% = ", mcmh_object.icdf[i][round(epsilon*cdf_sample)],mcmh_object.icdf[i][round((1-epsilon)*cdf_sample)-1],"98% = ", "95% = ")
#
#     print("Epsilons 99 = ", mcmh_object.layerMinimums[i]-mcmh_object.icdf[i][round(epsilon*cdf_sample)],
#           mcmh_object.icdf[i][round((1-epsilon)*cdf_sample)-1] - mcmh_object.layerMaximums[i], "Epsilons 98 = ", "Epsilons 95 = " )
#
#     mcmh_object.graphGoodSetDist(i)
#


print("max epsilon left: ", max(mcmh_object.epsiloni_left))

print("max epsilon right: ", max(mcmh_object.epsiloni_right))












# PROBABLY  GOOD IDEA TO RECOMPUTE THE IPQs from files!!!! :

# HAVE TO CONSOLIDATE THE OUTPUT PROPERTY FILE WITH THE "x" part of the original property file!
# nnet_object1.getInputQuery(output_filename1,)



# solve_query(ipq, filename="", verbose=True, timeout=0, verbosity=2)







