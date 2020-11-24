#!/usr/bin/python3

import os
import site
import sys
import getopt
import datetime

# from tabulate import tabulate

if '/cs' in os.getcwd():
    REMOTE = True
    if 'MARABOU_DIR' in os.environ.keys():
        MARABOU_DIR = os.environ['MARABOU_DIR']
    else:
        MARABOU_DIR = './'
else:
    REMOTE = False
    MARABOU_DIR = './'

print(REMOTE)

if REMOTE:
    sys.path.append(MARABOU_DIR)
    os.chdir(MARABOU_DIR+'maraboupy')
else:
    os.chdir('/Users/alexus/coding/my_Marabou/Marabou/maraboupy')

print(sys.path)

print(os.getcwd())

# from MarabouNetworkNNetIPQ import *
# from MarabouNetworkNNetProperty import *

# from MarabouNetworkNNet import *

# from Marabou import *
# from MarabouNetworkNNetExtensions import *

# import MarabouCore

from maraboupy.CompositionalVerifier import *

# import re

import sys

# import parser


import time

import numpy as np

#import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
import scipy.stats as stats
from random import randint

start_time = time.time()


NETWORK = '1_4'
PROPERTY = '1'
LAYER = 5
if REMOTE:
    TIMEOUT = 100000
else:
    TIMEOUT = 600

PERFORM_INTERPOLANT_SEARCH = True

REDIRECT_OUTPUT = True
GUROBI = 'ON'

RETRIEVE_GUROBI_BOUNDS = True

VERIFY_ORIGINAL = False


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hsobrgn:t:l:p:",["network=","timeout=","layer=","property=", "no-search",
                                                                 "no-redirect", "verify-original", "no-gurobi"])
    except getopt.GetoptError:
        print('verify_interpolant.py --network=<network> --timeout=<timeout> --layer=<layer> --property=<property>')
        sys.exit(5)
    print(opts)
    print(args)
    for opt, arg in opts:
        if opt == '-h':
            print('verify_interpolant.py --network=<network> --timeout=<timeout> --layer=<layer> --property=<property>')
            sys.exit(0)
        elif opt in ('-n', "--network"):
            NETWORK = arg
        elif opt in ("-t", "--timeout"):
            TIMEOUT = int(arg)
        elif opt in ("-l", "--layer"):
            LAYER = int(arg)
        elif opt in ("-p", "--property"):
            PROPERTY = arg
        elif opt in ('-s', "--no-redirect"):
            REDIRECT_OUTPUT = False
        elif opt in ('-o', "--verify-original"):
            VERIFY_ORIGINAL = True
        elif opt in ('-r', "--no-search"):
            PERFORM_INTERPOLANT_SEARCH = False
        elif opt in ('-b', "--no-bounds"):
            RETRIEVE_GUROBI_BOUNDS = False
        elif opt in ('-g', "--no-gurobi"):
            GUROBI = 'OFF'
            # RETRIEVE_GUROBI_BOUNDS = False  # ??

current_date_formatted = datetime.datetime.today().strftime ('%d%m%Y')
stdout_file = MARABOU_DIR + 'cv_test_output_' + str(current_date_formatted) + '_' + NETWORK + '_prop_' + PROPERTY + \
              '_layer_' + str(LAYER) + '_gurobi_' + GUROBI + '_search='+ str(PERFORM_INTERPOLANT_SEARCH)

if REDIRECT_OUTPUT:
    sys.stdout = open(stdout_file, 'w')

network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_" + NETWORK + ".nnet"
print('\nNetwork: ACASXU_experimental_v2a_' + NETWORK + '.nnet\n')
property_filename = "../resources/properties/acas_property_" + PROPERTY+ ".txt"
print('Property: acas_property_' + PROPERTY + '.txt\n')
print("Interpolant search on layer", LAYER, "\n")

property_filename1 = "../resources/properties/acas_property_1.txt"

network_filename1 = "test/ACASXU_experimental_v2a_"+NETWORK+"layer_"+str(LAYER)+"_input.nnet"
network_filename2 = "test/ACASXU_experimental_v2a_"+NETWORK+"layer_"+str(LAYER)+"_output.nnet"

output_property_file = "output_property_test1_"+NETWORK+"_layer_"+str(LAYER)+".txt"
input_property_file = "input_property_" + "acas" + NETWORK + "_prop" + PROPERTY + "_layer_" + str(LAYER) + "_test1.txt"

disjunct_network_file = "test/ACASXU_experimental_v2a_"+NETWORK+"_layer_"+str(LAYER)+"_disjunct.nnet"

mcmh_object = CompositionalVerifier(network_filename=network_filename, property_filename=property_filename, layer=LAYER)

mcmh_object.prepareForMarabouCandidateVerification(network_filename1=network_filename1,
                                                   network_filename2=network_filename2,
                                                   property_filename1=output_property_file,
                                                   property_filename2=input_property_file,
                                                   network_filename_disjunct=disjunct_network_file)
mcmh_object.initiateVerificationProcess(N=5000, compute_loose_offsets='range')

test_split_network(mcmh_object.marabou_nnet, mcmh_object.nnet_object1, mcmh_object.nnet_object2, layer=LAYER)


current_time = time.time()
if VERIFY_ORIGINAL:
    print("\nVerifying the original query with Marabou for comparison.\n")
    ipq = MarabouCore.InputQuery()
    MarabouCore.createInputQuery(ipq, network_filename, property_filename)
    options = Marabou.createOptions(verbosity=0)
    Marabou.solve_query(ipq,options=options)

marabou_time = time.time() - current_time

print('Marabou time: ', marabou_time)

start_time = time.time()

# mcmh_object.layer_interpolant_candidate.setInitialParticipatingNeurons(zero_bottoms=False)

# mcmh_object.createOriginalOutputPropertyFile()
# mcmh_object.addLayerPropertiesToOutputPropertyFile()

# mcmh_object.createOriginalInputPropertyFile()




# mcmh_object.addOutputLayerPropertyByIndexToInputPropertyFile(13, 'r')
#
# mcmh_object.createTruncatedInputNetworkByNeuron(var=49)

# ipq = MarabouCore.InputQuery()
#
# MarabouCore.createInputQuery(ipq, network_filename, property_filename)

# for var in range(5):
#     print(ipq.getLowerBound(var), mcmh_object.ipq.getUpperBound(var))
#
# print('hidden layer: ')
# for var in range(5, 10):
#     print(ipq.getLowerBound(var), mcmh_object.ipq.getUpperBound(var))
#
# print(mcmh_object.layer_interpolant_candidate.list_of_neurons[14].deltas['l'],
#       mcmh_object.layer_interpolant_candidate.list_of_neurons[14].deltas['r'],
#       mcmh_object.layer_interpolant_candidate.list_of_neurons[14].tight_bounds,
#       mcmh_object.layer_interpolant_candidate.list_of_neurons[14].suggested_bounds['l'],
#       mcmh_object.layer_interpolant_candidate.list_of_neurons[14].suggested_bounds['r'],
#       '\n',
#       mcmh_object.layer_interpolant_candidate.getConjunction())

# included_vars = choices(range(mcmh_object.layer_size),k=10)
#
# for var in range(mcmh_object.layer_size):
#     for side in types_of_bounds:
#         if var not in included_vars:
#             include_side = False
#         else:
#             include_side = choice([False, True])
#
#         if include_side:
#             mcmh_object.layer_interpolant_candidate.includeInInvariant(var, side)
#         else:
#             mcmh_object.layer_interpolant_candidate.excludeFromInvariant(var,side)
#

counter = 0
conjunction_verified = False

# print(mcmh_object.layer_interpolant_candidate.layer_minimums)
# print(mcmh_object.layer_interpolant_candidate.layer_maximums)

# for var in range(mcmh_object.basic_statistics.layer_size):
#     mcmh_object.basic_statistics.graphGoodSetDist(var)

print("\nPerforming randomized candidate search. \n")

for i in range(5):
    current_time = time.time()

    results = mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=5,
                                           verbosity=3, extremes=False)

    print(results)

    print('time check conjunction took: ', time.time() - current_time)

    if results[0] == 'success':
        break

for i in range(5):
    current_time = time.time()

    results = mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=5,
                                           verbosity=3, extremes=True)

    print(results)

    print('time check conjunction took: ', time.time() - current_time)

    if results[0] == 'success':
        break


print("\nInterpolant candidate discovered:\n")
for var in range(mcmh_object.layer_interpolant_candidate.layer_size):
    print(mcmh_object.layer_interpolant_candidate.list_of_neurons[var].getSuggestedLowerBound(),
          '<= x' + str(var) + '<= ',
          mcmh_object.layer_interpolant_candidate.list_of_neurons[var].getSuggestedUpperBound())
    
    
    
if RETRIEVE_GUROBI_BOUNDS:
    current_time = time.time()

    print("\nRetrieving Gurobi bounds for comparison.\n")
    ipq = MarabouCore.InputQuery()
    MarabouCore.createInputQuery(ipq, network_filename, property_filename)
    # options = Marabou.createOptions(verbosity=2)
    MarabouCore.preprocess(ipq)
    print('Preprocessing done.')
    print('Preprocessing time: ', time.time() - current_time)
    print('\nComparing the bounds.\n')
    print('Layer = ', mcmh_object.layer)
    print('\n\n')
    print("{:<15} {:<42} {:<20} {:<20} {:<20} {:<20} {:<20}".format('node',
                                                                    'Gurobi lower',
                                                                    'Candidate real lower',
                                                                    'Candidate lower',
                                                                    'Candidate upper',
                                                                    'Candidate real upper',
                                                                    'Gurobi upper'))
    print("{:<15} {:<42} {:<20} {:<20} {:<20} {:<20} {:<20}".format('____',
                                                                    '______________________',
                                                                    '______________________',
                                                                    '______________________',
                                                                    '______________________',
                                                                    '______________________',
                                                                    '______________________'))
    for var in range(mcmh_object.layer_interpolant_candidate.layer_size):
        node = mcmh_object.marabou_nnet.nodeTo_b(mcmh_object.layer,var)
        print("{:<15} {:<42} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
            'node: '+str(var),
            ipq.getLowerBound(node),
            mcmh_object.layer_interpolant_candidate.list_of_neurons[var].real_bounds_for_invariant['l'],
            mcmh_object.layer_interpolant_candidate.list_of_neurons[var].getSuggestedLowerBound(),
            mcmh_object.layer_interpolant_candidate.list_of_neurons[var].getSuggestedUpperBound(),
            mcmh_object.layer_interpolant_candidate.list_of_neurons[var].real_bounds_for_invariant['r'],
            ipq.getUpperBound(node)))

    if not PERFORM_INTERPOLANT_SEARCH:
        print('Not performing candidate verification.')
        sys.exit(0)

# marabou_time = time.time() - current_time
#
# print('Marabou time: ', marabou_time)

# print('\n Bpounds discovered by Gurobi: ')
#     for var in range(mcmh_object.layer_interpolant_candidate.layer_size):
#         print()

# bad_input, _ = mcmh_object.verifyConjunctionWithMarabou(add_to_badset=True)
#
# current_time = time.time()
#
# if bad_input:
#     mcmh_object.detailedDumpLayerInput(bad_input)
#     out_of_bounds_inputs, differene_dict = \
#         mcmh_object.layer_interpolant_candidate.analyzeBadLayerInput(bad_input)
#
#     print(out_of_bounds_inputs)
#     print(differene_dict)
#
#     epsilon_adjusted = mcmh_object.adjustConjunctionOnBadInput(bad_input, adjust_epsilons='random',
#                                                                number_of_epsilons_to_adjust=10)
#
#     print(epsilon_adjusted)
#
#     print('Time Marabou took: ', time.time() - current_time)
#

print("\nVerifying conjunction with Marabou.\n")
while (time.time() - start_time < TIMEOUT):
    counter += 1
    current_time = time.time()

    bad_input, _ = mcmh_object.verifyConjunctionWithMarabou(add_to_badset=True)

    conjunction_time = time.time() - current_time

    print("conjuction verification time: ", conjunction_time)

    if bad_input:
        mcmh_object.detailedDumpLayerInput(bad_input)
        out_of_bounds_inputs, differene_dict = \
            mcmh_object.layer_interpolant_candidate.analyzeBadLayerInput(bad_input)

        print(out_of_bounds_inputs)
        print(differene_dict)

        epsilon_adjusted = mcmh_object.adjustConjunctionOnBadInput(bad_input, adjust_epsilons='half_all')

        print(epsilon_adjusted)
        if not epsilon_adjusted.keys():
            print('Nothing adjusted, seems like we are stuck for now.')
            print('Moving to phase 2')
            break
    else:
        print('success')
        print('number of loops: ', counter)
        conjunction_verified = True
        break

# if not conjunction_verified:
#     print('failure')
#     print('number of loops: ', counter)
#     sys.exit(0)

if conjunction_verified:
    print('\nOriginal conjunction modified and verified. Time elapsed', time.time() - start_time)
else:
    print('\nOriginal conjunction modified but not verified. Time elapsed', time.time() - start_time)

# mcmh_object.layer_interpolant_candidate.excludeFromInvariant(var=14, side='r')
#
# new_start_time = time.time()
#
# conjunction_verified = False
# counter = 0
# while (time.time() - new_start_time < 600):
#     counter += 1
#     current_time = time.time()
#
#     print(mcmh_object.layer_interpolant_candidate.getConjunction())
#     MarabouCore.createInputQuery(mcmh_object.ipq2, mcmh_object.network_filename2, mcmh_object.property_filename2)
#     print(mcmh_object.ipq2.getUpperBound(14))
#     print(mcmh_object.ipq2.getLowerBound(14))
#
#     bad_input, _ = mcmh_object.verifyConjunctionWithMarabou(add_to_badset=True)
#     print(mcmh_object.ipq2.getUpperBound(14))
#     print(mcmh_object.ipq2.getLowerBound(14))
#
#     conjunction_time = time.time() - current_time
#
#     print("conjunction verification time: ", conjunction_time)
#
#     if bad_input:
#         mcmh_object.detailedDumpLayerInput(bad_input)
#         out_of_bounds_inputs, differene_dict = \
#             mcmh_object.layer_interpolant_candidate.analyzeBadLayerInput(bad_input)
#
#         print(out_of_bounds_inputs)
#         print(differene_dict)
#
#         epsilon_adjusted = mcmh_object.adjustConjunctionOnBadInput(bad_input, adjust_epsilons='random',
#                                                                    number_of_epsilons_to_adjust=5)
#
#         print(epsilon_adjusted)
#     else:
#         print('success')
#         print('number of loops: ', counter)
#         conjunction_verified = True
#         break
#
#
#
# if not conjunction_verified:
#     print('failure')
#     print('number of loops: ', counter)
#     sys.exit(0)
#
# print('new conjunction modified and verified. Time elapsed', time.time() - new_start_time)
#

new_start_time = time.time()

print("\nVerifying disjunction with Marabou.\n")



failed_disjuncts, exit_due_to_timeout = mcmh_object.verifyUnverifiedDisjunctsWithMarabou(add_to_goodset=False,
                                                                                         timeout=TIMEOUT, verbosity=2,
                                                                                         truncated_output_layer=False)


current_time = time.time()

print('Disjuncts time: ', current_time-new_start_time)
print('Total time so far: ', current_time - start_time)


if exit_due_to_timeout:
    print('time out!')
    sys.exit(0)

if not failed_disjuncts:
    print('No failed disjuncts')
    if conjunction_verified:
        print("Interpolant candidate search has succeeded.")
        print("\n\nMaking sure this is indeed an invariant: \n")
        print('\nVerifying all the disjuncts: \n')
        new_start_time = time.time()
        failed_disjuncts = mcmh_object.verifyAllDisjunctsWithMarabou(truncated_output_layer=False, verbosity=3)
        if failed_disjuncts:
            warnings.warn('Failed disjuncts still present!')
            print('Something went wrong! There are ', len(failed_disjuncts), ' failed disjuncts.')
            print([(x[0],x[1],mcmh_object.layer_interpolant_candidate[x[0]].suggested_bounds[x[1]],
                    mcmh_object.layer_interpolant_candidate[x[0]].verified_disjunct[x[1]]) for x in failed_disjuncts])
            failed_disjuncts, _ = mcmh_object.verifyUnverifiedDisjunctsWithMarabou(truncated_output_layer=False)
            print(len(failed_disjuncts), 'unverified disjuncts have also failed: ')
            print([(x[0],x[1]) for x in failed_disjuncts])

        else:
            print('All disjuncts clear out.')

        sys.exit(0)
else:
    print('Number of failed disjuncts: ', len(failed_disjuncts))
    print('failed disjuncts: ', failed_disjuncts)
    print("Interpolant candidate search has failed.")

new_start_time = time.time()


print('Adjusting disjuncts and continuing in a loop.')

status, argument_list = mcmh_object.CandidateSearch(number_of_trials=1000, individual_sample=100, verbosity=3,
                                                    timeout=TIMEOUT, truncated_output_layer=False)

current_time = time.time()

search_time = current_time - start_time
print('Disjuncts time: ', current_time - new_start_time)
print('Total search time: ', search_time)

if status == 'success':
    print('UNSAT')
    print('Interpolant for layer = ', mcmh_object.layer, ':')
    print(mcmh_object.layer_interpolant_candidate.getConjunction())

    print("\n\nMaking sure this is indeed an invariant: \n")
    print('\nVerifying all the disjuncts: \n')
    new_start_time = time.time()
    failed_disjuncts = mcmh_object.verifyAllDisjunctsWithMarabou(truncated_output_layer=False, verbosity=3)
    if failed_disjuncts:
        warnings.warn('Failed disjuncts still present!')
        print('Something went wrong! There are ', len(failed_disjuncts), ' failed disjuncts.')
        print([(x[0],x[1],mcmh_object.layer_interpolant_candidate[x[0]].suggested_bounds[x[1]],
                mcmh_object.layer_interpolant_candidate[x[0]].verified_disjunct[x[1]]) for x in failed_disjuncts])
        failed_disjuncts, _ = mcmh_object.verifyUnverifiedDisjunctsWithMarabou(truncated_output_layer=False)
        print(len(failed_disjuncts), 'unverified disjuncts have also failed: ')
        print([(x[0],x[1]) for x in failed_disjuncts])

    else:
        print('All disjuncts clear out.')

    print("\nVerifying the conjunction: \n")
    mcmh_object.verifyConjunctionWithMarabou(add_to_badset=False)
    print('Gurobi is ', GUROBI)
    print('Original direct marabou verification time: ', marabou_time)
    print('Minimal safety margin: ', mcmh_object.layer_interpolant_candidate.minimal_safety_margin)
    print('Total candidate search time: ', search_time)
    print('Total candidate verification time: ', time.time()-new_start_time)
    sys.exit(0)

if status == 'raw_conjunction_too_weak':
    print('Interpolant search has failed. A counterexample found between observed layer bounds. Check if SAT?')
    print('Bad input is ', argument_list)
    sys.exit(2)

if status == 'timeout':
    print('Timeout after ', argument_list[0], 'seconds.')
    sys.exit(3)



sys.exit(4)

# status, list =


counter = 0

while time.time() - new_start_time < TIMEOUT:
    counter += 1
    mcmh_object.adjustDisjunctsOnBadInputs(failed_disjuncts=failed_disjuncts)
    failed_disjuncts, exit_due_to_timeout = mcmh_object.verifyUnverifiedDisjunctsWithMarabou(add_to_goodset=False,

                                                                                            timeout=TIMEOUT, verbosity=2,
                                                                                             truncated_output_layer=False)
    print('Disjunct rounds number ', counter)

    current_time = time.time()
    print('Disjuncts time so far: ', current_time - new_start_time)
    print('Total time so far: ', current_time - start_time)

    if exit_due_to_timeout:
        print('time out!')
        sys.exit(0)

    if not failed_disjuncts:
        print('No failed disjuncts')
        print("Interpolant candidate search has succeeded.")
        sys.exit(0)
    else:
        print('Number of failed disjuncts: ', len(failed_disjuncts))
        print('failed disjuncts: ', failed_disjuncts)
        print("Interpolant candidate search has failed.")



print("Time out!")
sys.exit(0)

print("\nVerifying the original query with Marabou for comparison.\n")

ipq = MarabouCore.InputQuery()
MarabouCore.createInputQuery(ipq, network_filename, property_filename)
Marabou.solve_query(ipq)

print('Marabou time: ', time.time() - current_time)

sys.exit(0)

# network_filename = "../maraboupy/regress_acas_nnet/ACASXU_run2a_1_7_batch_2000.nnet"


mcmh_object = MarabouNNetMCMH(network_filename=network_filename, property_filename=property_filename)
mcmh_object.marabou_nnet.property.compute_executables()

# solve_query(mcmh_object.marabou_query.ipq2,property_filename)

# print(nnet_object.marabou_query.property.exec_bounds)
# print(nnet_object.marabou_query.property.exec_equations)

# nnet_object.outputsOfInputExtremes()

mcmh_object.setLayer(layer=5)

mcmh_object.createInitialGoodSet(N=10000, adjust_bounds=True, sanity_check=False)

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


mcmh_object.splitNetwork(output_filename1, output_filename2)

# Testing the random output property file!
# nnet_object1 = MarabouNetworkNNetExtended(output_filename1,output_property_file1)

nnet_object2 = MarabouNetworkNNetExtended(output_filename2, input_property_file)

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

print("Time taken: ", time.time() - start_time)

# test_split_network(mcmh_object.marabou_nnet,nnet_object1,nnet_object2)
#

mcmh_object.computeStatistics()
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


for i in range(mcmh_object.marabou_nnet.layerSizes[mcmh_object.layer]):
    print("variable: ", i)
    print("mean = ", mcmh_object.mean[i])
    print("median = ", mcmh_object.median[i])
    print("sigma = ", mcmh_object.sigma[i])
    print("sigma left = ", mcmh_object.sigma_left[i])
    print("sigma right = ", mcmh_object.sigma_right[i])

    print("sigma m left= ", mcmh_object.msigma_left[i])
    print("sigma m right = ", mcmh_object.msigma_right[i])

    print("sigma = ", mcmh_object.sigma[i])

    print("Observed bounds: ", mcmh_object.layerMinimums_dict[i], mcmh_object.layerMaximums_dict[i])

    print("Bounds based on 3 sigma left and right:",
          mcmh_object.mean[i] - 3 * mcmh_object.sigma_left[i], mcmh_object.mean[i] + 3 * mcmh_object.sigma_right[i])

    print("Bounds based on 3 m sigma left and right:",
          mcmh_object.median[i] - 3 * mcmh_object.msigma_left[i],
          mcmh_object.median[i] + 3 * mcmh_object.msigma_right[i])

    print("Bounds based on 4 sigma left and right:",
          mcmh_object.mean[i] - 3.5 * mcmh_object.sigma_left[i], mcmh_object.mean[i] + 3.5 * mcmh_object.sigma_right[i])

    print("Bounds based on 4 m sigma left and right:",
          mcmh_object.median[i] - 3.5 * mcmh_object.msigma_left[i],
          mcmh_object.median[i] + 3.5 * mcmh_object.msigma_right[i])

    print("Bounds computed with epsilon left and right: ",
          mcmh_object.layerMinimums_dict[i] - mcmh_object.epsiloni_left[i],
          mcmh_object.layerMaximums_dict[i] + mcmh_object.epsiloni_right[i])

    print("range: ", mcmh_object.range[i])

    print("Epsilons: ", mcmh_object.epsiloni[i], mcmh_object.epsiloni_left[i], mcmh_object.epsiloni_right[i])

    mcmh_object.graphGoodSetDist(i)

print("max epsilon left: ", max(mcmh_object.epsiloni_left))

print("max epsilon right: ", max(mcmh_object.epsiloni_right))


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = stats.gaussian_kde(x, bw_method='scott', **kwargs)
    return kde


x_grid = np.linspace(-4.5, 3.5, 1000)

good_set_array = np.array(mcmh_object.good_set)

print(good_set_array[:, 17])

kde = kde_scipy(good_set_array[:, 17], x_grid)

kde1 = kde.evaluate(x_grid.tolist())

kde_list = mcmh_object.kde_eval

# PROBABLY  GOOD IDEA TO RECOMPUTE THE IPQs from files!!!! :

# HAVE TO CONSOLIDATE THE OUTPUT PROPERTY FILE WITH THE "x" part of the original property file!
# nnet_object1.getInputQuery(output_filename1,)


# solve_query(ipq, filename="", verbose=True, timeout=0, verbosity=2)
