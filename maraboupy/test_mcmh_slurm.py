#!/usr/bin/python3

import os
import sys
#os.chdir('/cs/labs/guykatz/alexus/my_marbou/Marabou-1/maraboupy')
sys.path.append('/cs/labs/guykatz/alexus/my_marbou/Marabou-1/maraboupy/')

print (os.getcwd())
#from MarabouNetworkNNetIPQ import *
#from MarabouNetworkNNetProperty import *

from MarabouNetworkNNetExtended import *

from Marabou import *
from MarabouNetworkNNetExtentions import *

from MarabouNNetMCMH import *

# import re

# import parser


import time

import numpy as np

#import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
import scipy.stats as stats
from random import randint


sys.stdout = open('./test_mcmh_slurm_output', 'w')



start_time = time.time()

network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet"
property_filename = "../resources/properties/acas_property_4.txt"
property_filename1 = "../resources/properties/acas_property_1.txt"


network_filename1 = "test/ACASXU_experimental_v2a_1_9_output1.nnet"
network_filename2 = "test/ACASXU_experimental_v2a_1_9_output2.nnet"


output_property_file = "output_property_test1.txt"
input_property_file = "input_property_test1.txt"


mcmh_object = MarabouNNetMCMH(network_filename=network_filename, property_filename=property_filename, layer=5)

mcmh_object.initiateVerificationProcess(N=5000,compute_loose_offsets='range')
mcmh_object.prepareForMarabouCandidateVerification(network_filename1=network_filename1, network_filename2=network_filename2,
                                                   property_filename1=output_property_file, property_filename2=input_property_file)


mcmh_object.layer_interpolant_candidate.setInitialParticipatingNeurons(zero_bottoms=True)


'''
print(mcmh_object.layer_interpolant_candidate.list_of_neurons[14].deltas['l'],
      mcmh_object.layer_interpolant_candidate.list_of_neurons[14].deltas['r'],
      mcmh_object.layer_interpolant_candidate.list_of_neurons[14].tight_bounds,
      mcmh_object.layer_interpolant_candidate.list_of_neurons[14].suggested_bounds['l'],
      mcmh_object.layer_interpolant_candidate.list_of_neurons[14].suggested_bounds['r'],
      '\n',
      mcmh_object.layer_interpolant_candidate.getConjunction())
'''

counter = 0
conjunction_verified = False

current_time = time.time()

print(mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=5,
                                   verbosity=2,extremes=False))

print('time check conjunction took: ', time.time() - current_time)

current_time = time.time()


print(mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=5,
                                   verbosity=2,extremes=True))

print('time check conjunction took: ', time.time() - current_time)


while(time.time()-start_time<600):
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

        epsilon_adjusted = mcmh_object.adjustConjunctionOnBadInput(bad_input,adjust_epsilons='half_all')

        print(epsilon_adjusted)
    else:
        print('success')
        print('number of loops: ', counter)
        conjunction_verified = True
        break


if not conjunction_verified:
    print('failure')
    print('number of loops: ', counter)
    sys.exit(0)

print('original conjunction modified and verified. Time elapsed', time.time() - start_time)

sys.exit(0)

mcmh_object.layer_interpolant_candidate.excludeFromInvariant(var=14, side='r')

new_start_time = time.time()

conjunction_verified = False
counter = 0
while(time.time()-new_start_time<600):
    counter += 1
    current_time = time.time()


    print(mcmh_object.layer_interpolant_candidate.getConjunction())
    MarabouCore.createInputQuery(mcmh_object.ipq2, mcmh_object.network_filename2, mcmh_object.property_filename2)
    print(mcmh_object.ipq2.getUpperBound(14))
    print(mcmh_object.ipq2.getLowerBound(14))


    bad_input, _  = mcmh_object.verifyConjunctionWithMarabou(add_to_badset=True)
    print(mcmh_object.ipq2.getUpperBound(14))
    print(mcmh_object.ipq2.getLowerBound(14))



    conjunction_time = time.time() - current_time

    print("conjuction verification time: ", conjunction_time)

    if bad_input:
        mcmh_object.detailedDumpLayerInput(bad_input)
        out_of_bounds_inputs, differene_dict = \
        mcmh_object.layer_interpolant_candidate.analyzeBadLayerInput(bad_input)

        print(out_of_bounds_inputs)
        print(differene_dict)

        epsilon_adjusted = mcmh_object.adjustConjunctionOnBadInput(bad_input,adjust_epsilons='random',
                                                                   number_of_epsilons_to_adjust=5)

        print(epsilon_adjusted)
    else:
        print('success')
        print('number of loops: ', counter)
        conjunction_verified = True
        break

if not conjunction_verified:
    print('failure')
    print('number of loops: ', counter)
    sys.exit(0)

print('new conjunction modified and verified. Time elapsed', time.time() - new_start_time)
sys.exit(0)



failed_disjuncts, exit_due_to_timeout = mcmh_object.verifyUnverifiedDisjunctsWithMarabou(add_to_goodset=False,
                                                                                         timeout=600, verbosity=2)

if exit_due_to_timeout:
    print('time out!')

if not failed_disjuncts:
    print('No failed disjuncts')
else:
    print('failed disjuncts: ', failed_disjuncts)


sys.exit(0)



#network_filename = "../maraboupy/regress_acas_nnet/ACASXU_run2a_1_7_batch_2000.nnet"


mcmh_object = MarabouNNetMCMH(network_filename=network_filename, property_filename=property_filename)
mcmh_object.marabou_nnet.property.compute_executables()

# solve_query(mcmh_object.marabou_nnet.ipq2,property_filename)

#print(nnet_object.marabou_nnet.property.exec_bounds)
#print(nnet_object.marabou_nnet.property.exec_equations)

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

print("Time taken: ",time.time()-start_time)

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
    print("mean = ",mcmh_object.mean[i])
    print("median = ",mcmh_object.median[i])
    print("sigma = ",mcmh_object.sigma[i])
    print("sigma left = ",mcmh_object.sigma_left[i])
    print("sigma right = ",mcmh_object.sigma_right[i])

    print("sigma m left= ",mcmh_object.msigma_left[i])
    print("sigma m right = ",mcmh_object.msigma_right[i])

    print("sigma = ",mcmh_object.sigma[i])

    print("Observed bounds: ", mcmh_object.layerMinimums_dict[i], mcmh_object.layerMaximums_dict[i])

    print("Bounds based on 3 sigma left and right:",
          mcmh_object.mean[i]-3*mcmh_object.sigma_left[i], mcmh_object.mean[i]+3*mcmh_object.sigma_right[i])

    print("Bounds based on 3 m sigma left and right:",
          mcmh_object.median[i]-3*mcmh_object.msigma_left[i], mcmh_object.median[i]+3*mcmh_object.msigma_right[i])

    print("Bounds based on 4 sigma left and right:",
          mcmh_object.mean[i]-3.5*mcmh_object.sigma_left[i], mcmh_object.mean[i]+3.5*mcmh_object.sigma_right[i])

    print("Bounds based on 4 m sigma left and right:",
          mcmh_object.median[i]-3.5*mcmh_object.msigma_left[i], mcmh_object.median[i]+3.5*mcmh_object.msigma_right[i])

    print("Bounds computed with epsilon left and right: ",
          mcmh_object.layerMinimums_dict[i] - mcmh_object.epsiloni_left[i], mcmh_object.layerMaximums_dict[i] + mcmh_object.epsiloni_right[i])

    print("range: ", mcmh_object.range[i])

    print("Epsilons: ", mcmh_object.epsiloni[i],mcmh_object.epsiloni_left[i],mcmh_object.epsiloni_right[i])

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


print(good_set_array[: , 17])

kde = kde_scipy(good_set_array[: , 17], x_grid)

kde1 = kde.evaluate(x_grid.tolist())


kde_list =mcmh_object.kde_eval












# PROBABLY  GOOD IDEA TO RECOMPUTE THE IPQs from files!!!! :

# HAVE TO CONSOLIDATE THE OUTPUT PROPERTY FILE WITH THE "x" part of the original property file!
# nnet_object1.getInputQuery(output_filename1,)



# solve_query(ipq, filename="", verbose=True, timeout=0, verbosity=2)







