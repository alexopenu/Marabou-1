
#from MarabouNetworkNNetIPQ import *
#from MarabouNetworkNNetProperty import *

from MarabouNetworkNNetExtended import *

from Marabou import *
import MarabouCore
from MarabouNetworkNNetExtentions import *

# import re
# import parser

import sys
import time

import numpy as np

from random import choice
from random import choices

import seaborn as sns
import matplotlib.pyplot as plt

# import matplotlib.mlab as mlab
# import scipy.stats as stats
# from statsmodels.nonparametric.api import KDEUnivariate
# from random import randint




# 'l' stands for left (lower)
# 'r' stands for right (upper)
types_of_bounds = ['l','r']

class basic_mcmh_statistics:
    def __init__(self,good_set = [],epsilon=0.01,two_sided = True):
        self.epsilon = epsilon

        self.layer_size = 0
        self.good_matrix = np.array([])
        self.minimums = []
        self.maximums = []
        self.mean = []
        self.epsiloni = []
        self.epsiloni_twosided = {'l': [], 'r': []}
        self.range = []

        self.statistics_computed = False

        if good_set:
            self.recompute_statistics(good_set,two_sided)


    def recompute_statistics(self,good_set=[],two_sided=True):

        if not good_set:
            self.statistics_computed = False
            self.layer_size = 0
            self.good_matrix = np.array([])
            self.minimums = []
            self.maximums = []
            self.mean = []
            self.epsiloni = []
            self.epsiloni_twosided = {'l': [], 'r': []}
            self.range = []
        else:
            self.good_matrix = np.array(good_set)
            self.layer_size = len(good_set[0])
            self.minimums = [np.min(self.good_matrix[:,var]) for var in range(self.layer_size)]
            self.maximums = [np.max(self.good_matrix[:, var]) for var in range(self.layer_size)]
            self.mean = [np.mean(self.good_matrix[:, var]) for var in range(self.layer_size)]

            self.computeEpsilonsUniformly(two_sided)
            self.statistics_computed = True



    def computeEpsilonsUniformly(self):
        '''
        computes epsilons (to be added to or subtracted from empiric upper and lower bounds for the hidden layer
            variables) based on a very simple statistical heuristics, essentially assuming that the distibution
            on the values of each variable is close to uniform. This is not the case, but the computation is
            efficient, and seems to give "conservative" bounds, which should be a good starting point.
        :return:
        '''

        assert self.good_matrix
        for var in range(self.layer_size):
            self.computeEpsilonsForVariable(var)


    def computeEpsilonsForVariable(self,var,compute_twosided = True):

        outputs = self.good_matrix[:, var]
        sample_size = len(outputs)

        minimum = self.minimums[var]
        maximum = self.maximums[var]
        mean = self.mean[var]

        self.range[var] = maximum-minimum
        self.epsiloni[var] = self.range[var]/sample_size

        if compute_twosided:
            small_outputs = [output for output in outputs if (output <= mean)]
            small_range = mean-minimum
            big_outputs = [output for output in outputs if (output > mean)]
            big_range = maximum-mean

            self.epsiloni_twosided['l'][var] = small_range/len(small_outputs)
            self.epsiloni_twosided['r'][var] = big_range/len(big_outputs)


    def graphGoodSetDist(self,i=0):

        assert self.statistics_computed

        sns.distplot([output[i] for output in self.good_matrix],label='Variable'+str(i))
        plt.legend()
        plt.show()




class MarabouNNetMCMH:

    def __init__(self, network_filename, property_filename, layer=-1):
        self.marabou_nnet = MarabouNetworkNNetExtended(filename=network_filename, property_filename=property_filename)

        self.network_filename = network_filename
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

        # Lists of the original input (x) and output (y) properties to be verified
        # (exactly as they appear in the property files)
        self.x_properties = []
        self.y_properties = []

        # Setting x_properties and y_properties to their correct values
        self.setOriginalPropertyLists()

        # List of properties (interpolant candidate) for the chosen layer
        # The properties are recorded as equations/bounds on input variables and stored as strings
        self.interpolant_candidate = []

        # List that contains the interpolant properties recorded as equations/bounds on output variables
        # is supposed to be identical to interpolant_candidate except that the inequalities are flipped
        # (and all occurrences of 'x' is replaced with 'y')
        self.output_interpolant_candidate = []

        self.dict_sidevar_to_interpolant_index = dict()

        self.good_set = []
        self.bad_set = []


        #Lests of the variables corresponding to the layer
        self.layer_bVars = []
        self.layer_fVars = []

        # various statistics and empiric bounds on layer variables for guessing the invariant
        # will be reset in seLayer, but we include all the attributes in the constructor
        self.good_matrix = np.array([])
        self.epsilon = 0.01
        self.epsiloni = []
        self.epsiloni_twosided = {'l': [], 'r': []}

        self.basic_statistics = basic_mcmh_statistics(epsilon=self.epsilon)

        # Attributes representing minimal and maximal values seen for the variables of the layer
        # The default ones are for the b-variables
        # Will be reset in setLayer

        self.layer_minimums = []
        self.layer_maximums= []
        self.layer_fminimums_dict = dict() #For sanity check
        self.layer_fmaximums_dict = dict() #For sanity check

        self.suggested_layer_bounds = {'l': [], 'r': []}
        # self.suggested_lower_bounds = []
        # self.suggested_upper_bounds = []


        # The hidden layer we will study an invariant for
        # Currently we only work with one hidden layer
        if layer>-1:
            self.setLayer(layer)
        else:
            self.layer = -1
            self.layer_size = 0



        # we store executable versions of certain properties (that will be evaluated many times) locally
        # for the sake of efficiency
        assert self.marabou_nnet.property.exec_equations_computed
        assert self.marabou_nnet.property.exec_bounds_computed

        self.input_equations = self.marabou_nnet.property.get_exec_input_equations()
        self.output_equations = self.marabou_nnet.property.get_exec_output_equation()
        self.output_bounds = self.marabou_nnet.property.get_exec_output_bounds()



        # The number of inputs that have been considered: used for computation of statistics
        # Not clear we need this at all. Check.
        self.numberOfInputs = 0

        # Filenames for the new networks created by splitting
        self.network_filename1 = ''
        self.network_filename2 = ''
        self.property_filename1 = ''
        self.property_filename2 = ''

        # Input queries reserved for verification of the interpolant candidate with Marabou
        self.ipq1 = MarabouCore.InputQuery()
        self.ipq2 = MarabouCore.InputQuery()



    def setLayer(self,layer):
        assert ((layer>=0) and (layer<self.marabou_nnet.numLayers))
        self.layer = layer
        self.computeLayerVariables()

        self.good_set = []
        self.layer_minimums = []
        self.layer_maximums = []

        self.layer_fminimums_dict = {}
        self.layer_fmaximums_dict = {}

        self.basic_statistics = basic_mcmh_statistics(epsilon=self.epsilon)

        # self.mean = {}
        # self.sigma = {}
        # self.epsiloni = {}
        # self.epsiloni_left = {}
        # self.epsiloni_right = {}


    def setEpsilon(self, epsilon: float):
        self.epsilon = epsilon


    def setFilenames(self,property_filename1='',property_filename2=''):
        assert property_filename1
        assert property_filename2

        self.property_filename1=property_filename1
        self.property_filename2=property_filename2

    def setOriginalPropertyLists(self):
        if self.marabou_nnet.property.mixed_properties_present() or self.marabou_nnet.property.ws_properties_present():
            print('Currently only support pure input and output properties')
            sys.exit(1)

        self.x_properties = self.marabou_nnet.property.get_original_x_properties()
        self.y_properties = self.marabou_nnet.property.get_original_y_properties()

    def computeLayerVariables(self):
        self.layer_bVars = []
        self.layer_fVars = []

        layer = self.layer

        self.layer_size = self.marabou_nnet.layerSizes[layer]
        for node in range(self.layer_size):
            self.layer_fVars.append(self.marabou_nnet.nodeTo_f(layer, node))
            self.layer_bVars.append(self.marabou_nnet.nodeTo_b(layer, node))


    def clearGoodSet(self):
        self.good_set = []


    def verifyInputEquations(self, x):
        for eq in self.input_equations:
            if not eval(eq):
                return False
        return True

    def verifyOutputBounds(self, y):
        for eq in self.output_bounds:
            if not eval(eq):
                return False
        return True

    def verifyOutputEquations(self, y):
        for eq in self.output_equations:
            if not eval(eq):
                return False
        return True

    def verifyOutputProperty(self, y):
        return self.verifyOutputBounds(y) and self.verifyOutputEquations(y)


    def outputVariableToIndex(self,output_variable):
        return output_variable-self.marabou_nnet.numVars+self.marabou_nnet.outputSize


    def initiateVerificationProcess(self,N=5000):
        assert self.layer>-1

        self.createInitialGoodSet(N,include_input_extremes=True,adjust_bounds=False)
        assert self.good_set

        self.basic_statistics.recompute_statistics(self.good_set,two_sided=True)
        assert self.basic_statistics.statistics_computed

        self.epsiloni = self.basic_statistics.epsiloni
        self.epsiloni_twosided = self.basic_statistics.epsiloni_twosided
        self.layer_minimums = self.basic_statistics.minimums
        self.layer_maximums = self.basic_statistics.maximums

        self.computeInitialInterpolantCandidateForLayer()


    def computeInitialInterpolantCandidateForLayer(self,two_sided=True):

        assert self.layer_minimums
        assert self.layer_maximums

        if two_sided:
            assert self.epsiloni_twosided['l']
            assert self.epsiloni_twosided['r']
        else:
            assert self.epsiloni

        self.suggested_layer_bounds['l'] = [self.getSoftLowerBoundForLayer(var,two_sided) for var in range(self.layer_size)]
        self.suggested_layer_bounds['r'] = [self.getSoftUpperBoundForLayer(var,two_sided) for var in range(self.layer_size)]

        self.recomputeInterpolant()

    def recomputeInterpolant(self):
        assert self.suggested_layer_bounds['l']
        assert self.suggested_layer_bounds['r']

        self.interpolant_candidate = []
        self.output_interpolant_candidate = []

        for var in range(self.layer_size):
            lb=self.computeLowerBoundProperty(var)
            ub=self.computeUpperBoundProperty(var)
            self.interpolant_candidate.append({'l': lb,'r': ub})

            if self.suggested_layer_bounds['r'][var] == 0:
                ub_y = ''
            else:
                ub_y = ub.replace('x','y').replace('>','<')
            lb_y = lb.replace('x','y').replace('<','>')
            self.output_interpolant_candidate.append({'l': lb_y,'r': ub_y})

    def adjustInterpolantCandidate(self,var: int, side = '', two_sided = True):
        assert var in range(self.layer_size)
        assert side in types_of_bounds

        self.suggested_layer_bounds[side][var] = self.getSoftBoundForLayer(var,side,two_sided)

        x_bdd = self.computeBoundProperty(var,side)

        if side == 'r' and self.suggested_layer_bounds['r'][var] == 0:
            y_bdd = ''
        else:
            y_bdd = x_bdd.replace('x','y').replace('<','>').replace('>','<')

        self.interpolant_candidate[var][side] = x_bdd
        self.output_interpolant_candidate[var][side] = y_bdd



    def adjustEpsilon(self, var: int, side: str, new_epsilon: float, two_sided = True, adjust_candidate = True):
        assert var in range(self.layer_size)
        assert side in types_of_bounds

        if not two_sided:
            assert self.epsiloni
            assert len(self.epsiloni) > var
            self.epsiloni[var] = new_epsilon
            return

        assert self.epsiloni_twosided[side]
        assert len(self.epsiloni_twosided[side])>var
        self.epsiloni_twosided[side][var] = new_epsilon

        if adjust_candidate:
            self.adjustInterpolantCandidate(var,side,two_sided)


    def computeLowerBoundProperty(self,var: int):
        return 'x' + str(var) + ' <= ' + str(self.suggested_layer_bounds['l'][var])

    def computeUpperBoundProperty(self,var: int):
        return 'x' + str(var) + ' >= ' + str(self.suggested_layer_bounds['r'][var])

    def computeBoundProperty(self,var: int, side: str):
        assert side in types_of_bounds

        if side == 'l':
            return self.computeLowerBoundProperty(var)
        return self.computeUpperBoundProperty(var)


    def getSoftUpperBoundForLayer(self, var, two_sided = True):
        epsilon = self.epsiloni_twosided['r'][var] if two_sided else self.epsiloni[var]
        if not epsilon:
            epsilon = self.epsilon
        return max(0, self.layer_maximums[var] + epsilon)


    def getSoftLowerBoundForLayer(self, var, two_sided = True):
        epsilon = self.epsiloni_twosided['l'][var] if two_sided else self.epsiloni[var]
        if not epsilon:
            epsilon = self.epsilon
        return max(0, self.layer_minimums[var] - epsilon)

    def getSoftBoundForLayer(self, var: int, side: str, two_sided = True):
        assert side in types_of_bounds

        if side == 'l':
            return self.getSoftLowerBoundForLayer(var,two_sided)
        return self.getSoftUpperBoundForLayer(var,two_sided)


    def adjustInitialCandidate(self,total_trials = 100, individual_sample = 20, number_of_epsilons_to_adjust = 1,
                               timeout=0):

        assert self.interpolant_candidate

        out_of_bound_inputs = []
        difference_dict = {}

        starting_time = time.time()

        for i in range(total_trials):
            if timeout>0:
                time_elapsed = time.time() - starting_time
                if timeout and time_elapsed > timeout:
                    return 'timeout', out_of_bound_inputs, difference_dict, time_elapsed

            result,  epsilon_adjusted, out_of_bounds_inputs, difference_dict =  \
                self.checkConjunction(sample_size=individual_sample, adjust_epsilons=True, add_to_bad_set=True,
                                      number_of_epsilons_to_adjust=number_of_epsilons_to_adjust)
            if result:
                return 'success', [], {}, 0

        return 'failed', out_of_bound_inputs, difference_dict, 0


    def checkConjunction(self, sample_size=100, adjust_epsilons = True, add_to_bad_set=True,
                         number_of_epsilons_to_adjust = 1):
        assert self.interpolant_candidate
        status = True # No counterexample found yet

        out_of_bounds_inputs = []
        difference_dict = {}
        epsilon_adjusted = []


        for i in range(sample_size):
            layer_input = self.createRandomSoftInputsForLayer()

            result, one_out_of_bounds_inputs, one_differene_dict, output = \
                self.checkCandidateOnInput(layer_input, add_to_bad_set=add_to_bad_set)
            # Note that we do not adjust epsilons on one input here;
            # rather, we will later choose an epsilon (or epsilons) to adjust at random
            # from the whole bad set of inputs after the loop is finished


            if result == 'within_bounds': # The counterexample is between the actual observed bounds
                print("Candidate search has failed. Found a counterexample between observed lower and "
                      "upper bounds: \n layer input = ",layer_input, '\n output = ', output,   "\n Check if SAT?")
                sys.exit(2)

            if result == 'out_of_bounds':  # A counterexample to the candidate found
                status = False
                out_of_bounds_inputs += one_out_of_bounds_inputs

                # adjusting the maximal difference dictionary
                for (var,side) in one_differene_dict.keys():
                    if (var,side) not in difference_dict.keys() \
                            or difference_dict[(var,side)]<one_differene_dict[(var,side)]:
                        difference_dict[(var,side)] = one_differene_dict[(var,side)]

        if (not status) and adjust_epsilons:
            epsilon_adjusted = self.strengthenEpsilons(out_of_bounds_inputs,difference_dict,adjust_epsilons='random',
                                    number_of_epsilons=number_of_epsilons_to_adjust)

        # if (not status) and adjust_epsilons:
        #     self.adjustEpsilonAtRandom(out_of_bounds_inputs, adjust_candidate=True)

        return status, epsilon_adjusted, out_of_bounds_inputs, difference_dict





    def checkCandidateOnInput(self, layer_input,  add_to_bad_set = True):
        # assert adjust_epsilons in ['','random','all']

        output = self.marabou_nnet.evaluateNetworkFromLayer(layer_input, first_layer=self.layer)
        one_out_of_bounds_inputs = []
        one_differene_dict = {}

        result = 'success' # No counterexamples to the candidate found

        if not self.verifyOutputProperty(y=output):
            if add_to_bad_set:
                self.bad_set.append(layer_input)

            one_out_of_bounds_inputs, one_differene_dict = self.analyzeBadLayerInput(layer_input)

            if not one_out_of_bounds_inputs: # Bad input within observed bounds!
                result = 'within_bounds'
            else:
                result  = 'out_of_bounds'

        return result, one_out_of_bounds_inputs, one_differene_dict, output


    def strengthenEpsilons(self, out_of_bounds_inputs, differene_dict, adjust_epsilons='', number_of_epsilons=1):
        assert adjust_epsilons in ['', 'random', 'all', 'half_all','half_random']
        assert number_of_epsilons>=0

        if not adjust_epsilons:
            return []

        weights=[difference for (var,side,difference) in out_of_bounds_inputs]
        if adjust_epsilons == 'random' or adjust_epsilons == 'half_random':
            epsilons_to_adjust = choices(out_of_bounds_inputs,weights=weights,k=number_of_epsilons)
        else:
            epsilons_to_adjust = out_of_bounds_inputs

        for (var,side,_) in epsilons_to_adjust:
            if adjust_epsilons == 'half_all' or adjust_epsilons == 'half_random':
                new_epsilon = self.epsiloni_twosided[side][var] / 2
            elif adjust_epsilons == 'all' or adjust_epsilons == 'random'
                new_epsilon = self.epsiloni_twosided[side][var](1 + differene_dict[(var, side)]) / 2
            else:
                print('Strange error in strengthenEpsilons')
                sys.exit(1)

            self.adjustEpsilon(var,side,new_epsilon=new_epsilon,two_sided=True,adjust_candidate=True)

        return epsilons_to_adjust




    def analyzeBadLayerInput(self, layer_input, use_multiplicity = False):
        bad_layer_inputs = []
        bad_layer_inputs_dict = {}

        for var in range(self.layer_size):
            if layer_input[var] < self.layer_minimums[var]:  # lb - delta[i,'l'] <= xi <= lb (lb = layer minimum)
                side = 'l'
                difference = (self.layer_minimums[var] - layer_input[var])
            elif layer_input[var] > self.layer_maximums[var] and self.suggested_layer_bounds['r'][var] > 0:
                side = 'r'
                difference = (layer_input[var] - self.layer_maximums[var])  # ub <= xi <= ub + delta[i,'r']
            else:
                continue

            difference = difference / self.epsiloni_twosided[side][var]
            if use_multiplicity:
                multiplicity = round(difference * 5)  # A PARAMETER THAT CAN BE ADJUSTED!
                bad_layer_inputs += [(var, side, difference)] * multiplicity
            else:
                bad_layer_inputs += [(var, side, difference)]

            if (var,side) not in bad_layer_inputs_dict.keys() or bad_layer_inputs_dict[(var,side)]<difference:
                bad_layer_inputs_dict[(var,side)] = difference

        return bad_layer_inputs, bad_layer_inputs_dict



    def initiateCandidateSearch(self, total_trials=100, individual_sample = 20, timeout = 0):

        starting_time = time.time()

        result, out_of_bounds_inputs, differences_dict, time_elapsed = \
            self.adjustInitialCandidate(total_trials=total_trials, individual_sample=individual_sample,timeout=timeout)

        # if result == 'success':
        #     status = 'success'
        #     return status, result, [], time.time() - starting_time

        if result == 'timeout':
            status = 'candidate_too_weak'
            return status, result, [], time.time()-starting_time

        if result == 'failed':
            bad_input = self.verifyRawConjunction() # Verifying whether the observed bounds are strong enough
            if bad_input:
                print("Search for interpolant candidate failed.\nBad input within empirical bounds found by "
                      "Marabou: ", bad_input)
                sys.exit(2)
            status = 'candidate_too_weak'
            result = 'checking_conjunction_failed'
            return status, result, [], time.time()-starting_time

        # if time.time()-starting_time>timeout:
        #     status = 'candidate_too_weak'
        #     result = 'timeout'
        #     return status, result, [], time.time()-starting_time

        bad_input = self.verifyConjunctionWithMarabou(add_to_badset=True)

        if bad_input:
            print('Bad input found by Marabou: ', bad_input)
            status = 'candidate_too_weak'
            result = 'verification_conjunction_failed'
            return status, result, bad_input, time.time()-starting_time

        if timeout and time.time()-starting_time>timeout:
            status = 'candidate_not_too_weak'
            result = 'timeout'
            return status, result, [], time.time()-starting_time

        # So for now we assume that we have a candidate which is not too weak
        # Next step is verifying the disjunction

        failed_disjuncts = self.verifyAllDisjunctsWithMarabou(add_to_goodset=True)

        if not failed_disjuncts: # FOR NOW WE EXIT IF THE DISJUNCTION IS VERIFIED
            status = 'success'
            result = 'disjuncts_verified'
            return status, result, [], time.time()-starting_time
            # print('We are done!')
            # print(self.interpolant_candidate)
            # sys.exit(0)

        status = 'failed_disjuncts'
        result = 'disjuncts_verification_failed'
        return status, result, failed_disjuncts, time.time()-starting_time



    def CandidateSearch(self, number_of_trials: int, individual_sample: int, timeout = 0):

        starting_time = time.time()
        status, result, argument_list, time_elapsed = self.initiateCandidateSearch(total_trials=number_of_trials,
                                                                                   individual_sample = individual_sample,
                                                                                   timeout = timeout)
        if status == 'success':
            print('We are done!')
            print(self.interpolant_candidate)
            sys.exit(0)

        if timeout and time.time() - starting_time > timeout:
            return 'timeout', time.time() - starting_time

        if status == 'candidate_too_weak':
            if argument_list:
                bad_input = argument_list
                self.adjustEpsilonsOnLayerInput(layer_input=bad_input,adjust_epsilons='all')
            else:
                self.HalfBadEpsilonsOnBadInput()

            # TO DO: MODIFY!

        if status == 'candidate_not_too_weak':

        if status == 'failed_disjuncts':

            # COMPLETE!!!!




    # NOTE: creates full output file, not a full input file
    def initiateMarabouCandidateVerification(self, network_filename1: str, network_filename2: str, property_filename1: str,
                                        property_filename2: str):

        self.split_network(network_filename1, network_filename2)
        self.setFilenames(property_filename1,property_filename2)
        self.createOriginalInputPropertyFile()
        self.createOriginalOutputPropertyFile()
        self.addLayerPropertiesToOutputPropertyFile()


    def createOriginalInputPropertyFile(self):
        property_filename1 = self.property_filename1
        # assert property_filename1

        try:
            with open(property_filename1,'w') as f2:
                for l in self.x_properties:
                    f2.write(l)
        except:
            print('Something went wrong with creating the initial property files or writing to them')
            sys.exit(1)


    def createOriginalOutputPropertyFile(self):
        property_filename2 = self.property_filename2
        # assert property_filename2

        try:
            with open(property_filename2, 'w') as f2:
                for l in self.y_properties:
                    f2.write(l)
        except:
            print('Something went wrong with creating the initial property files or writing to them')
            sys.exit(1)


    def addLayerPropertiesToOutputPropertyFile(self):
        '''
        adds self.interpolant_candidate to self.property_filename2
            which is the property file for the network whose input layer is self.layer
        :return:
        '''
        # assert self.property_filename2

        try:
            with open(self.property_filename2,'a') as f2:
                for p in self.interpolant_candidate:
                    f2.write(p)
        except:
            print('Something went wrong with writing to property_file2')
            sys.exit(1)


    def addOutputLayerPropertyByIndexToInputPropertyFile(self,var: int, side: str):
        '''
        adds one property (string) from self.output_interpolant_candidate to self.property_filename1
            which is the property file for the network whose output layer is self.layer
        note that what needs to be verified for this network is the disjunction of self.output_interpolant_candidate
            (hence it makes sense to add one at a time)
        :return:
        '''

        assert var in range(self.layer_size)
        assert side in types_of_bounds

        self.addOutputLayerPropertyToInputPropertyFile(p=self.output_interpolant_candidate[var][side])

    def addOutputLayerPropertyToInputPropertyFile(self,p=''):
        '''
        adds one property (string) to self.property_filename1
            which is the property file for the network whose output layer is self.layer
        note that what needs to be verified for this network is the disjunction of self.output_interpolant_candidate
            (hence it makes sense to add one at a time)
        :return:
        '''

        # assert self.property_filename1

        try:
            with open(self.property_filename1, 'a') as f2:
                f2.write(p)
        except:
            print('Something went wrong with writing to property_file1')
            sys.exit(1)


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



    # def recompute_xPropertyFile(self):
    #     property_filename2 = self.property_filename2
    #     assert property_filename2


    # def recomputePropertyFiles(self):
    #     property_filename1 = self.property_filename1
    #     property_filename2 = self.property_filename2


    # Currently there is no use for this, and the dictionary is empty
    def getPropertyIndex(self,var,side):
        assert (var,side) in self.dict_sidevar_to_interpolant_index
        return self.dict_sidevar_to_interpolant_index[(var,side)]




    def verifyConjunctionWithMarabou(self,add_to_badset=True):
        MarabouCore.createInputQuery(self.ipq1,self.network_filename1,self.property_filename1)

        [vals,_] = solve_query(ipq=self.ipq1,verbose=True,verbosity=1)
        bad_input = self.convertVectorFromDictToList(vals)
        if bad_input:  # SAT
            if add_to_badset:
                self.good_set.append(bad_input)
        return bad_input


    def verifyDisjunctWithMarabou(self,,var: int, side: str, add_to_goodset=True):
        assert var in range(self.layer_size)
        assert side in types_of_bounds

        self.createOriginalOutputPropertyFile()
        self.addOutputLayerPropertyByIndexToInputPropertyFile(var, side)
        MarabouCore.createInputQuery(self.ipq2, self.network_filename2, self.property_filename2)

        [vals, _] = solve_query(self.ipq2, verbosity=1)
        bad_input = self.convertVectorFromDictToList(vals)
        if bad_input:  # SAT
            if add_to_goodset:
                self.good_set.append(bad_input)
        return bad_input


    def verifyAllDisjunctsWithMarabou(self,add_to_goodset=True):
        failed_disjuncts = []
        for var in range(self.layer_size):
            for side in types_of_bounds:
                bad_input = self.verifyDisjunctWithMarabou(var,side,add_to_goodset=add_to_goodset)
                if not bad_input: #UNSAT
                    continue
                failed_disjuncts.append((var,side,bad_input))

        return failed_disjuncts


    def convertVectorFromDictToList(self,dict_vector: dict):
        return [dict_vector[i] for i in dict_vector.keys()]



    def createInitialGoodSet(self, N, include_input_extremes=True, adjust_bounds=False, sanity_check=False):
        self.clearGoodSet()
        if include_input_extremes:
            self.outputsOfInputExtremesForLayer(adjust_bounds=adjust_bounds,add_to_goodset=True,add_to_statistics=True,
                                                verify_property=True,sanity_check=sanity_check)
        self.addRandomValuesToGoodSet(N,adjust_bounds=adjust_bounds,check_bad_inputs=True,sanity_check=sanity_check)




    def createRandomInputs(self):
        input = []
        for input_var in self.marabou_nnet.inputVars.flatten():
            # Note: the existence of lower and upper bounds was asserted in the constructor
            random_value = np.random.uniform(low=self.marabou_nnet.lowerBounds[input_var],
                                             high=self.marabou_nnet.upperBounds[input_var])
            input.append(random_value)
        return input



    def createRandomHardInputsForLayer(self):
        input = []
        for var in range(self.layer_size):
            random_value = np.random.uniform(low=self.layer_minimums[var],
                                             high=self.layer_maximums[var])
            input.append(random_value)
        return input


    def createRandomSoftInputsForLayer(self):
        input = []
        for var in range(self.layer_size):
            random_value = np.random.uniform(low=self.suggested_layer_bounds['l'][var],
                                             high=self.suggested_layer_bounds['r'][var])
            input.append(random_value)
        return input


    def adjustLayerBounds(self,layer_output):
        """
        Adjust lower and upper bounds for the layer
        :param layer_output: list of floats
        :return:
        """
        for i in range(self.layer_size):
            if (not (i in self.layer_minimums)) or (layer_output[i] < self.layer_minimums[i]):
                self.layer_minimums[i] = layer_output[i]
            if (not (i in self.layer_maximums)) or (layer_output[i] > self.layer_maximums[i]):
                self.layer_maximums[i] = layer_output[i]


    def adjustfLayerBounds(self,layer_output):
        """
        Adjust lower and upper bounds for the f-variables of the layer
        Currently used only for sanity check
        :param layer_output: list of floats
        :return:
        """
        for i in range(self.layer_size):
            if (not (i in self.layer_fmaximums_dict)) or (layer_output[i] < self.layer_fminimums_dict[i]):
                self.layer_fminimums_dict[i] = layer_output[i]
            if (not (i in self.layer_fmaximums_dict)) or (layer_output[i] > self.layer_fmaximums_dict[i]):
                self.layer_fmaximums_dict[i] = layer_output[i]



    def addRandomValuesToGoodSet(self,N,adjust_bounds=True, check_bad_inputs = True, sanity_check=False):
        assert self.layer >=0

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

    def verifyRawConjunction(self):

        self.createOriginalOutputPropertyFile()

        try:
            with open(self.property_filename2,'a') as f2:
                for var in range(self.layer_size):
                    p = 'x <= '+str(self.layer_maximums[var])
                    f2.write(p)
                    p = 'x >= '+str(self.layer_minimums[var])
                    f2.write(p)
        except:
            print('Something went wrong with writing to property_file2')
            sys.exit(1)

        MarabouCore.createInputQuery(self.ipq2,self.property_filename2,self.property_filename2)
        [vals,_] = solve_query(self.ipq2,verbosity=1)
        return self.convertVectorFromDictToList(vals)




    # Creates a list of outputs for self.layer for the "extreme" input values
    def outputsOfInputExtremesForLayer(self, adjust_bounds = True, add_to_goodset = True, add_to_statistics = True,
                                       verify_property = True, sanity_check = False):
        layer_outputs = []
        input_size = self.marabou_nnet.inputSize

        assert self.layer>-1

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
                    inputs[input_var] = self.marabou_nnet.upperBounds[input_var]
                else:
                    inputs[input_var] = self.marabou_nnet.lowerBounds[input_var]

            # print ("Evaluating layer; input = ", inputs) # debug

            #Evaluating the network up to the given layer on the input
            #By not activating the last layer, we get values for the b variables, which give more information

            output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=self.layer,normalize_inputs=False,
                                                              normalize_outputs=False,activate_output_layer=False)

            # print("output = ", output) # debug
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



    # Older version, probably redundant?
    # def createAdjustEpsilonDict(self, out_of_bound_inputs):
    #     adjust_epsilons_dict = {}
    #
    #     for current_var, current_side, current_difference in out_of_bound_inputs:
    #         if ((current_var,current_var) not in adjust_epsilons_dict.keys()) \
    #                 or (adjust_epsilons_dict[(current_var,current_side)<current_difference]):
    #             adjust_epsilons_dict[(current_var,current_side)] = current_difference
    #
    #     return adjust_epsilons_dict
    #
    #
    # def HalfBadEpsilonsOnBadInput(self, one_out_of_bounds_inputs, adjust_candidate=True):
    #     for current_var, current_side, current_epsilon in one_out_of_bounds_inputs:
    #         new_epsilon = self.epsiloni_twosided[current_side][current_var]/2
    #         self.adjustEpsilon(current_var, current_side, new_epsilon, two_sided=True, adjust_candidate=adjust_candidate)
    #
    #
    #
    # def adjustAllEpsilons(self, out_of_bounds_inputs, adjust_candidate=True):
    #     adjust_epsilons_dict = self.createAdjustEpsilonDict(out_of_bounds_inputs)
    #
    #     for current_var, current_side in adjust_epsilons_dict.keys()
    #
    #         max_difference = adjust_epsilons_dict[(current_var,current_side)]
    #
    #         new_epsilon = self.epsiloni_twosided[current_side][current_var](1 + max_difference) / 2
    #
    #         self.adjustEpsilon(current_var, current_side, new_epsilon, two_sided=True, adjust_candidate=adjust_candidate)
    #
    #
    # def adjustEpsilonAtRandom(self, out_of_bounds_inputs, adjust_candidate=True):
    #     (random_var, random_side, random_difference) = choice(out_of_bounds_inputs)
    #
    #     max_difference = max([difference for (var, side, difference) in out_of_bounds_inputs
    #                           if (var == random_var and side == random_side)])
    #
    #     new_epsilon = self.epsiloni_twosided[random_side][random_var](1 + max_difference) / 2
    #
    #     self.adjustEpsilon(random_var, random_side, new_epsilon, two_sided=True, adjust_candidate=adjust_candidate)
    #
    # def adjustEpsilonsOnLayerInput(self, layer_input, adjust_epsilons=''):
    #     assert adjust_epsilons in ['', 'random', 'all', 'half_all']
    #
    #     one_out_of_bounds_inputs = []
    #
    #     if adjust_epsilons == 'half_all':
    #         for var in range(self.layer_size):
    #             if layer_input[var] < self.layer_minimums[var]:  # lb - delta[i,'l'] <= xi <= lb (lb = layer minimum)
    #                 side = 'l'
    #             elif layer_input[var] > self.layer_maximums[var] and self.suggested_layer_bounds['r'][var] > 0:
    #                 side = 'r'
    #             else:
    #                 continue
    #
    #             one_out_of_bounds_inputs += [(var, side, 0)]
    #
    #         self.HalfBadEpsilonsOnBadInput(one_out_of_bounds_inputs, adjust_candidate=True)
    #     else:
    #         for var in range(self.layer_size):
    #             if layer_input[var] < self.layer_minimums[var]:  # lb - delta[i,'l'] <= xi <= lb (lb = layer minimum)
    #                 side = 'l'
    #                 difference = (self.layer_minimums[var] - layer_input[var])
    #             elif layer_input[var] > self.layer_maximums[var] and self.suggested_layer_bounds['r'][var] > 0:
    #                 side = 'r'
    #                 difference = (layer_input[var] - self.layer_maximums[var])  # ub <= xi <= ub + delta[i,'r']
    #             else:
    #                 continue
    #
    #             difference = difference / self.epsiloni_twosided[side][var]
    #             if adjust_epsilons == 'random':
    #                 multiplicity = round(difference * 5)  # A PARAMETER THAT CAN BE ADJUSTED!
    #             else:
    #                 multiplicity = 1
    #
    #             one_out_of_bounds_inputs += [(var, side, difference)] * multiplicity
    #
    #         if adjust_epsilons == 'random':
    #             self.adjustEpsilonAtRandom(one_out_of_bounds_inputs, adjust_candidate=True)
    #         elif adjust_epsilons == 'all':
    #             self.adjustAllEpsilons(one_out_of_bounds_inputs, adjust_candidate=True)
    #
    #     return one_out_of_bounds_inputs
    #



    # Old version, redundant now
    # def createInitialPropertyFiles(self,property_filename1,property_filename2):
    #     '''
    #     Creates property files for the two networks based on the original property
    #     properties involving x only are copied to property_filename1
    #     properties involving y only are copied to property_filename2
    #     the new filenames are stored in self.property_filename1 and self.property_filename2
    #     if self.property_filename1 or self.property_filename2 are not empty, throws an exception
    #     if a property that involves 'ws' (a hidden variable) or a 'mixed' property (i.e., involving both
    #         x and y) is found, throws an exception
    #
    #     :param property_filename1: str
    #     :param property_filename2: str
    #     :return:
    #     '''
    #     assert not self.property_filename1
    #     assert not self.property_filename2
    #
    #     x_properties = []
    #     y_properties = []
    #
    #     for p in self.marabou_nnet.property.properties_list:
    #         if p['type2'] == 'x':
    #             x_properties.append(p['line'])
    #         elif p['type2'] == 'y':
    #             y_properties.append(p['line'])
    #         else:
    #             print('Only pure input and output properties are currently supported')
    #             sys.exit(1)
    #
    #     try:
    #         with open(property_filename1,'w') as f2:
    #             for l in x_properties:
    #                 f2.write(l)
    #         with open(property_filename2,'w') as f2:
    #             for l in y_properties:
    #                 f2.write(l)
    #     except:
    #         print('Something went wrong with creating the initial property files or writing to them')
    #         sys.exit(1)
    #
    #     # self.create_xPropertyFile(property_filename1)
    #     # self.create_yPropertyFile(property_filename2)
    #
    #     self.property_filename1 = property_filename1
    #     self.property_filename2 = property_filename2
    #
    #





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



    # Old version, probablly redundant
    # def create_new_xPropertyFile(self,property_filename1):
    #
    #     assert self.property_filename1
    #
    #     print('Warning: replacing property_filename1, from ', self.property_filename1, " to ", property_filename1)
    #
    #     x_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'x']
    #
    #     try:
    #         with open(property_filename1,'w') as f2:
    #             for l in x_properties:
    #                 f2.write(l)
    #     except:
    #         print('Something went wrong with creating the initial property files or writing to them')
    #         sys.exit(1)
    #
    #     self.property_filename1 = property_filename1


    # Old version, probably redundant
    # def create_new_yPropertyFile(self, property_filename2):
    #
    #     assert self.property_filename2
    #
    #     print('Warning: replacing property_filename2, from ', self.property_filename2, " to ", property_filename2)
    #
    #     y_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'y']
    #
    #     try:
    #         with open(property_filename2, 'w') as f2:
    #             for l in y_properties:
    #                 f2.write(l)
    #     except:
    #         print('Something went wrong with creating the initial property files or writing to them')
    #         sys.exit(1)
    #
    #     self.property_filename2 = property_filename2


    # I think this method is redundant
    # def createPropertyFiles(self,property_filename1,property_filename2):
    #     # if self.marabou_nnet.property.bounds['ws'] or self.marabou_nnet.property.equations['m'] or
    #     #     self.marabou_nnet.property.equations['ws']:
    #     #     print('Mixed equations and bounds on hidden variables currently not supported')
    #     #     sys.exit(1)
    #
    #     x_properties = []
    #     y_properties = []
    #
    #     for p in self.marabou_nnet.property.properties_list:
    #         if p['type2'] == 'x':
    #             x_properties.append(p['line'])
    #         elif p['type2'] == 'y':
    #             y_properties.append(p['line'])
    #         else:
    #             print('Only pure input and output properties are currently supported')
    #             sys.exit(1)
    #
    #     # input_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'x']
    #     # output_properties = [p in self.marabou_nnet.property.properties_list for p['type2'] == 'y']
    #
    #     try:
    #         with open(property_filename1,'w') as f2:
    #             for l in x_properties:
    #                 f2.write(l)
    #         with open(property_filename2,'w') as f2:
    #             for l in y_properties:
    #                 f2.write(l)
    #     except:
    #         print('Something went wrong with creating the initial property files or writing to them')
    #         sys.exit(1)
    #
    #     self.property_filename1 = property_filename1
    #     self.property_filename2 = property_filename2
    #




    # REDUNDANT?
    # def computeInitialInterpolantCandidateForLayer(self,sanity_check = False):
    #     '''
    #     computes the initial interpolant candidate, based on the empiric bounds for the self.layer variables
    #         and on the epsilons computed using statistical analysis of the values of these variables
    #     the candidate itself is stored in self.interpolant_candidate (list of strings)
    #         each string is of the form "xi <= ??" or "xi >= ??"
    #         it is going to be used as the input property for the network whose input layer is self.layer
    #     the "dual" candidate - with x replaced with y and the inequalities reversed in stored in
    #         self.output_interpolant_candidate
    #     :return:
    #     '''
    #     if not sanity_check:
    #         layerMinimums = self.layer_minimums
    #         layerMaximums = self.layer_maximums
    #     else:
    #         layerMinimums = self.layerfMinimums
    #         layerMaximums = self.layerfMaximums
    #
    #     interpolant_list = []
    #     output_interpolant_list = []
    #
    #     for i in range(self.marabou_nnet.layerSizes[self.layer]):
    #         if i in layerMinimums:
    #             individual_property = []
    #             individual_property.append('x')
    #             individual_property.append(str(i))
    #             individual_property.append(" >= ")
    #             if i in self.epsiloni_:
    #                 epsilon = self.epsiloni_twosided['l'][i]
    #             else:
    #                 epsilon = self.epsilon
    #             lower_bound = max(layerMinimums[i]-epsilon,0.0)
    #             individual_property.append(str(lower_bound))
    #             individual_property.append("\n")
    #
    #             individual_property_string = ''.join(individual_property)
    #
    #             interpolant_list.append(individual_property_string)
    #
    #             output_interpolant_list.append(individual_property_string.replace('x','y').replace('>','<'))
    #         if i in layerMaximums:
    #             individual_property = []
    #             individual_property.append("x")
    #             individual_property.append(str(i))
    #             individual_property.append(" <= ")
    #             if i in self.epsiloni_right:
    #                 epsilon = self.epsiloni_twosided['r'][i]
    #             else:
    #                 epsilon = self.epsilon
    #             upper_bound = max(layerMaximums[i] + epsilon,0.0)
    #             # if layerMaximums[i]<0:
    #             #     upper_bound = 0.0
    #             # else:
    #             #     upper_bound = layerMaximums[i]+self.epsiloni_right[i]
    #             individual_property.append(str(upper_bound))
    #             individual_property.append("\n")
    #
    #             individual_property_string = ''.join(individual_property)
    #
    #             interpolant_list.append(individual_property_string)
    #
    #             output_interpolant_list.append(individual_property_string.replace('x','y').replace('<','>'))
    #
    #     self.interpolant_candidate = interpolant_list
    #     self.output_interpolant_candidate = output_interpolant_list
    #


    # This method is from an older version, and is now redundant
    # def createRandomOutputPropertyFileForLayer(self,output_property_filename: str):
    #     """
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
    #     Chooses a disjunct at random (as long as it makes for an "interesting" property)
    #     and writes just that one into the property file
    #
    #     :param ouput_property_filename: str (the property file to be written into)
    #     :return:
    #     """
    #     epsilon = self.epsilon
    #
    #     try:
    #         with open(output_property_filename, 'w') as f2:
    #
    #             while(True):
    #                 i = randint(0,self.marabou_nnet.layerSizes[self.layer])
    #                 boundary = randint(0,1)
    #
    #
    #                 # I believe it is correct now
    #
    #                 if boundary:
    #                     lower_bound = self.layer_minimums[i] - epsilon
    #                     if (i in self.layer_minimums) and (lower_bound > 0):
    #                         f2.write("y")
    #                         f2.write(str(i))
    #                         f2.write(" <= ")  # NEGATING the property!
    #                         f2.write(str(lower_bound))
    #                         f2.write("\n")
    #                         break;
    #                 else:
    #                     if (i in self.layer_maximums):
    #                         if self.layer_maximums[i]<0:
    #                             upper_bound = 0.0
    #                         else:
    #                             upper_bound = self.layer_maximums[i] + epsilon
    #                         f2.write("y")
    #                         f2.write(str(i))
    #                         f2.write(" >= ")  # NEGATING the property!
    #                         f2.write(str(upper_bound))
    #                         f2.write("\n")
    #                         break;
    #
    #             with open(self.property_filename, 'r') as f:
    #                 line = f.readline()
    #                 while (line):
    #                     if ('w' in line):
    #                         print("w_i in the property file, not supported")
    #                         sys.exit(1)
    #                     if 'x' in line:
    #                         if ('y' in line):
    #                             print("Both x and y in the same equation in the property file, exiting")
    #                             sys.exit(1)
    #                         else:
    #                             f2.write(line)
    #                     line = f.readline()
    #
    #     except:
    #         print("Something went wrong with writing to the output property file",output_property_filename)
    #         sys.exit(1)

    # This method is from an older version, and is now redundant
    # def createSingleOutputPropertyFileForLayer(self,output_property_filename: str, i: int, lb: bool):
    #     """
    #     Create a property filename for a network in which self.layer is the output layer
    #     Assumes that the layer is not activated
    #     Encodes the empiric lower and upper bounds for the b-variables of the layer, which
    #     are currently stored in the dictionaries self.layerMinimums and self.layerMaximums
    #
    #     NOTE that self.layer is a hidden layer, so only positive bounds matter
    #
    #     The names of the variables in the property file are going to be yi, where i is the index
    #     of the variable in the layer
    #
    #
    #     Creates a property file that encodes one disjunct, corresponding to:
    #     i (int): the index of the variable
    #     lb (bool): True means lower bound, False means upper bound
    #
    #
    #     :param ouput_property_filename: str (the property file to be written into)
    #
    #     :return (bool): True if there is an interesting property to prove, False otherwise
    #     """
    #     epsilon = self.epsilon
    #
    #     interesting_property = False
    #
    #     try:
    #         with open(output_property_filename, 'w') as f2:
    #
    #             if lb:  # Lower bound
    #                 lower_bound = self.layer_minimums[i] - epsilon
    #                 if (i in self.layer_minimums) and (lower_bound > 0):
    #                     f2.write("y")
    #                     f2.write(str(i))
    #                     f2.write(" <= ")  # NEGATING the property!
    #                     f2.write(str(lower_bound))
    #                     f2.write("\n")
    #
    #                     interesting_property = True
    #
    #             else:  # Upper bound
    #                 if (i in self.layer_maximums):
    #                     if self.layer_maximums[i]<0:
    #                         upper_bound = 0.0
    #                     else:
    #                         upper_bound = self.layer_maximums[i] + epsilon
    #                     f2.write("y")
    #                     f2.write(str(i))
    #                     f2.write(" >= ")  # NEGATING the property!
    #                     f2.write(str(upper_bound))
    #                     f2.write("\n")
    #
    #                     interesting_property = True
    #
    #             if interesting_property:
    #                 '''
    #                 Copying the property for the input variables from the original property file
    #                 '''
    #                 try:
    #                     with open(self.property_filename, 'r') as f:
    #                         line = f.readline()
    #                         while (line):
    #                             if ('w' in line):
    #                                 print("w_i in the property file, not supported")
    #                                 sys.exit(1)
    #                             if 'x' in line:
    #                                 if ('y' in line):
    #                                     print("Both x and y in the same equation in the property file, exiting")
    #                                     sys.exit(1)
    #                                 else:
    #                                     f2.write(line)
    #                             line = f.readline()
    #                 except:
    #                     print("Something went wrong with copying from the property file to the output property file",
    #                           output_property_filename)
    #                     sys.exit(1)
    #
    #
    #     except:
    #         print("Something went wrong with writing to the output property file",output_property_filename)
    #         sys.exit(1)
    #
    #     return(interesting_property)

    # This method is from an older version, and is now redundant
    # def createInputPropertyFileForLayer(self,input_property_filename: str, sanity_check=False):
    #     """
    #     Create a property filename for a network in which self.layer is the input layer
    #     Assumes that the previous layer has been activated
    #     Encodes the empiric lower and upper bounds for the f-variables of the layer, which
    #     are computed from the bounds currently stored in the dictionaries self.layerMinimums and self.layerMaximums
    #
    #     If the upper bound if negative, we change it to 0
    #     Same for lower bound
    #
    #     If the upper bound is positive, we add epsilon to it
    #     If the lower bound is positive, subtract epsilon
    #
    #     The names of the variables in the property file are going to be xi, where i is the index
    #     of the variable in the layer
    #
    #     :param input_property_filename (str): the property file to be written into
    #     :param sanity_check (bool): if is True, uses self.layerfMinimums and self.layerfMaximums, for comparison
    #     :return:
    #     """
    #
    #     if not sanity_check:
    #         layerMinimums = self.layer_minimums
    #         layerMaximums = self.layer_maximums
    #     else:
    #         layerMinimums = self.layerfMinimums
    #         layerMaximums = self.layerfMaximums
    #
    #     try:
    #         with open(input_property_filename, 'w') as f2:
    #
    #             for i in range(self.marabou_nnet.layerSizes[self.layer]):
    #                 if i in layerMinimums:
    #                     f2.write("x")
    #                     f2.write(str(i))
    #                     f2.write(" >= ")
    #                     if i in self.epsiloni_twosided['l']:
    #                         epsilon = self.epsiloni_twosided['l'][i]
    #                     else:
    #                         epsilon = self.epsilon
    #                     lower_bound = max(layerMinimums[i]-epsilon,0.0)
    #                     f2.write(str(lower_bound))
    #                     f2.write("\n")
    #                 if i in layerMaximums:
    #                     f2.write("x")
    #                     f2.write(str(i))
    #                     f2.write(" <= ")
    #                     if i in self.epsiloni_twosided['r']:
    #                         epsilon = self.epsiloni_twosided['r'][i]
    #                     else:
    #                         epsilon = self.epsilon
    #                     upper_bound = max(layerMaximums[i] + epsilon,0.0)
    #                     # if layerMaximums[i]<0:
    #                     #     upper_bound = 0.0
    #                     # else:
    #                     #     upper_bound = layerMaximums[i]+self.epsiloni_right[i]
    #                     f2.write(str(upper_bound))
    #                     f2.write("\n")
    #     except:
    #         print("Something went wrong with writing to property file2",input_property_filename)
    #         sys.exit(1)
    #
    #
    #     self.property_filename2 = input_property_file

    # This method is from an older version, and is now redundant
    # def createPropertyFilesForLayer(self,property_filename1: str, property_filename2: str, sanity_check=False):
    #     '''
    #     THIS IS WRONG, uses the wrong version of createOuputProperty
    #
    #
    #     :param property_filename1:
    #     :param property_filename2:
    #     :param sanity_check:
    #     :return:
    #     '''
    #
    #     assert self.property_filename
    #     assert property_filename1
    #     assert property_filename2
    #
    #     try:
    #         self.createInputPropertyFileForLayer(property_filename2,sanity_check)
    #         self.createOutputPropertyFileForLayer(property_filename1)
    #
    #         # Appending the input and the output properties from the original property file to the
    #         # input and the output property files, respectively
    #
    #         with open(self.property_filename, 'r') as f:
    #             with open(property_filename1, 'a') as f1:
    #                 with open(property_filename2, 'a') as f2:
    #                     line = f.readline()
    #                     while (line):
    #                         if 'x' in line:
    #                             if 'y' in line:
    #                                 print("Both x and y in the same equation in the property file, exiting")
    #                                 sys.exit(1)
    #                             else:
    #                                 f1.write(line)
    #                         else:
    #                             if 'y' in line:
    #                                 f2.write(line)
    #                             else:
    #                                 print("Found an equation in the property file for non input/output variables, exiting")
    #                                 sys.exit(1)
    #                         line = f.readline()
    #                 #
    #                 # for i in range(self.marabou_nnet.layerSizes[self.layer]):
    #                 #     if i in layerMinimums:
    #                 #         f2.write("x")
    #                 #         f2.write(str(i))
    #                 #         f2.write(" >= ")
    #                 #         lower_bound = max(layerMinimums[i], 0.0)
    #                 #         f2.write(str(lower_bound))
    #                 #         f2.write("\n")
    #                 #     if i in layerMaximums:
    #                 #         f2.write("x")
    #                 #         f2.write(str(i))
    #                 #         f2.write(" <= ")
    #                 #         upper_bound = max(layerMaximums[i], 0.0)
    #                 #         f2.write(str(upper_bound))
    #                 #          f2.write("\n")
    #
    #
    #
    #     except:
    #         print("Something went wrong with writing to one of the property files")
    #         sys.exit(1)
    #
    #
    #     self.property_filename1 = property_filename1
    #     self.property_filename2 = property_filename2
    #
    #
    #
    #





    # NOT USED, and currently wrong (checks the negation of the property on the outputs)
    # See the method outputsOfInputExtremesForLayer, implemented correctly

    #Creates a list of outputs of the "extreme" input values
    #Checks whether all these outputs are legal (within bounds and satisfy the equations)
    #Creates a list of "empiric bounds" for the output variables based on the results
    # def outputsOfInputExtremes(self):
    #     outputs = []
    #     input_size = self.marabou_nnet.inputSize
    #     output_lower_bounds = dict()
    #     output_upper_bounds = dict()
    #
    #
    #     #print 2 ** input_size
    #
    #     #We don't want to deal with networks that have a large input layer
    #     assert input_size < 20
    #
    #     for i in range(2 ** input_size):
    #         #turning the number i into a bit string of a specific length
    #         bit_string =  '{:0{size}b}'.format(i,size=input_size)
    #
    #         #print bit_string #debugging
    #
    #         inputs = [0 for i in range(input_size)]
    #
    #         # creating an input: a sequence of lower and upper bounds, determined by the bit string
    #         for input_var in self.marabou_nnet.inputVars.flatten():
    #             if bit_string[input_var] == '1':
    #                 assert self.marabou_nnet.upperBoundExists(input_var)
    #                 inputs[input_var] = self.marabou_nnet.upperBounds[input_var]
    #             else:
    #                 assert self.marabou_nnet.lowerBoundExists(input_var)
    #                 inputs[input_var] = self.marabou_nnet.lowerBounds[input_var]
    #
    #
    #         print("input = ", inputs)
    #
    #         # Evaluating the network on the input
    #
    #         output = self.marabou_nnet.evaluateNetworkToLayer(inputs,last_layer=-1,normalize_inputs=False,normalize_outputs=False)
    #         print("output = ", output)
    #         outputs.append(output)
    #
    #         # if self.outputOutOfBounds(output)[0]: #NOT Normalizing outputs!
    #         #     print('A counterexample found! input = ', inputs)
    #         #
    #         #     sys.exit()
    #         #
    #         #
    #         # if not self.marabou_nnet.property.verify_equations_exec(inputs,output): #NOT Normalizing outputs!
    #         #     print('A counterexample found! One of the extreme values. Vector = ',bit_string, '; input = ', inputs)
    #         #
    #         #     sys.exit()
    #         #
    #         #
    #         #
    #         # Updating the smallest and the largest ouputs for output variables
    #         for output_var in self.marabou_nnet.outputVars.flatten():
    #             output_var_index = self.outputVariableToIndex(output_var)
    #             if not output_var in output_lower_bounds or output_lower_bounds[output_var]>output[output_var_index]:
    #                 output_lower_bounds[output_var] = output[output_var_index]
    #             if not output_var in output_upper_bounds or output_upper_bounds[output_var]<output[output_var_index]:
    #                 output_upper_bounds[output_var] = output[output_var_index]
    #
    #
    #     #print len(outputs)
    #     print ("lower bounds = ", output_lower_bounds)
    #     print ("upper bounds = ", output_upper_bounds)
    #
    #     #print(outputs)



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


'''
class advanced_mcmh_statistics:
    def __init__(self,good_set = [],epsilon = 0.04):

        self.good_matrix = np.array(good_set)

        self.epsilon = epsilon
        self.epsiloni = []
        self.epsiloni_twosided = {'l': [], 'r': []}

        self.mean = dict{}
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

        if self.good_matrix:
            self.computeStatistics()

    def computeStatistics(self):

        assert self.good_matrix

        self.layerSize = len(self.good_matrix[0])

        self.layerMinimums = [min(self.good_matrix[:,var]) for var in range(self.layerSize)]
        self.layerMaximums = [max(self.good_matrix[:, var]) for var in range(self.layerSize)]

        for var in range(self.layerSize):
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
        self.epsiloni_twosided['l'][var] = (outputs[round(sample_size/2)]-outputs[0])*1/len(small_outputs)
        self.epsiloni_twosided['r'][var] = (-outputs[round(sample_size/2)+1]+outputs[-1])*1/len(big_outputs)

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
        sns.distplot([output[i] for output in self.good_matrix],label='Variable'+str(i))
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




start_time = time.time()

network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet"
property_filename = "../resources/properties/acas_property_4.txt"
property_filename1 = "../resources/properties/acas_property_1.txt"


#network_filename = "../maraboupy/regress_acas_nnet/ACASXU_run2a_1_7_batch_2000.nnet"


mcmh_object = MarabouNNetMCMH(network_filename=network_filename, property_filename=property_filename)
mcmh_object.marabou_nnet.property.compute_executables()

# solve_query(mcmh_object.marabou_nnet.ipq2,property_filename)

#print(nnet_object.marabou_nnet.property.exec_bounds)
#print(nnet_object.marabou_nnet.property.exec_equations)

# nnet_object.outputsOfInputExtremes()

mcmh_object.setLayer(layer=5)

mcmh_object.createInitialGoodSet(N=1000, adjust_bounds=True, sanity_check=False)

print(mcmh_object.layerMinimums_dict)
print(mcmh_object.layerMaximums_dict)


mcmh_object.outputsOfInputExtremesForLayer(adjust_bounds=True, add_to_goodset=True, sanity_check=False)
print(mcmh_object.layerMinimums_dict)
print(mcmh_object.layerMaximums_dict)

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


# print("max epsilon left: ", max(mcmh_object.epsiloni_left))
#
# print("max epsilon right: ", max(mcmh_object.epsiloni_right))
#
#










# PROBABLY  GOOD IDEA TO RECOMPUTE THE IPQs from files!!!! :

# HAVE TO CONSOLIDATE THE OUTPUT PROPERTY FILE WITH THE "x" part of the original property file!
# nnet_object1.getInputQuery(output_filename1,)



# solve_query(ipq, filename="", verbose=True, timeout=0, verbosity=2)







