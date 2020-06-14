#!/usr/bin/python3

import os
import sys
#os.chdir('/cs/labs/guykatz/alexus/my_marbou/Marabou-1/maraboupy')

cluster_run = True if str(os.getcwd()).count('labs/guykatz') else False


if cluster_run: # Running on the cluster
    sys.path.append('/cs/labs/guykatz/alexus/my_marbou/Marabou-1/maraboupy/')
    sys.stdout = open('./test_mcmh_script_output', 'w')


print(os.getcwd())


from MarabouNNetMCMH import *

import time

import numpy as np

from random import randint

# import scipy.stats as stats
# import re

# import os
# import parser

# from MarabouNetworkNNetIPQ import *
# from MarabouNetworkNNetProperty import *
#
# from MarabouNetworkNNetExtended import *
#
# from Marabou import *
# from MarabouNetworkNNetExtentions import *


def repeated_conjunction_verification(mcmh_object: MarabouNNetMCMH):
    start_time = time.time()
    counter = 0
    conjunction_verified = False

    for i in range(5):
        current_time = time.time()

        print(
            mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=10,
                                         verbosity=2, extremes=False))

        print('time random check conjunction took: ', time.time() - current_time)

    for i in range(5):
        current_time = time.time()

        print(
            mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=10,
                                         verbosity=2, extremes=True))

        print('time extremes check conjunction took: ', time.time() - current_time)

    # print(mcmh_object.checkConjunction(total_trials=100, individual_sample=10000, number_of_epsilons_to_adjust=10,
    #                                    verbosity=2, extremes=True))

    print('time check conjunction took: ', time.time() - start_time)

    for var in range(mcmh_object.layer_interpolant_candidate.layer_size):
        print(mcmh_object.layer_interpolant_candidate.list_of_neurons[var].getSuggestedLowerBound(),
              '<= x' + str(var) + '<= ',
              mcmh_object.layer_interpolant_candidate.list_of_neurons[var].getSuggestedUpperBound())

    while (time.time() - start_time) < timeout:
        counter += 1
        current_time = time.time()


        bad_input, timed_out = mcmh_object.verifyConjunctionWithMarabou(add_to_badset=True, timeout = timeout)

        conjunction_time = time.time() - current_time

        print("conjunction verification time: ", conjunction_time)

        if timed_out:
            print('conjunction verification timed out!')
            break

        if bad_input:
            mcmh_object.detailedDumpLayerInput(bad_input)
            out_of_bounds_inputs, differene_dict = \
                mcmh_object.layer_interpolant_candidate.analyzeBadLayerInput(bad_input)

            print(out_of_bounds_inputs)
            print(differene_dict)

            epsilon_adjusted = mcmh_object.adjustConjunctionOnBadInput(bad_input, adjust_epsilons='random',
                                                                       number_of_epsilons_to_adjust=10)

            print(epsilon_adjusted)
            continue
        else:
            print('success')
            print('number of loops: ', counter)
            conjunction_verified = True
            break
    return conjunction_verified, counter


if cluster_run:
    timeout = 9000
else:
    timeout = 60


start_time = time.time()


for i in [4,2,3,5,6,1]:

    network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_1_" + str(i) + ".nnet"
    property_filename = "../resources/properties/acas_property_4.txt"

    print('\n\n\n\nNetwork file: ', network_filename)
    print('Property file: ', property_filename)
    # property_filename1 = "../resources/properties/acas_property_1.txt"


    network_filename1 = "test/ACASXU_experimental_v2a_1_9_output1_a.nnet"
    network_filename2 = "test/ACASXU_experimental_v2a_1_9_output2_a.nnet"


    output_property_file = "output_property_test1_a.txt"
    input_property_file = "input_property_test1_a.txt"


    mcmh_object = MarabouNNetMCMH(network_filename=network_filename, property_filename=property_filename, layer=5)

    mcmh_object.initiateVerificationProcess(N=5000,compute_loose_offsets='range')
    mcmh_object.prepareForMarabouCandidateVerification(network_filename1=network_filename1, network_filename2=network_filename2,
                                                       property_filename1=output_property_file, property_filename2=input_property_file)


    mcmh_object.layer_interpolant_candidate.setInitialParticipatingNeurons(zero_bottoms=True)



    conjunction_verified, counter = repeated_conjunction_verification(mcmh_object)

    if not conjunction_verified:
        print('failure')
        print('number of loops: ', counter)
        continue

    print('original conjunction modified and verified. Time elapsed', time.time() - start_time)
    print('number of loops: ', counter)


    new_start_time = time.time()

    failed_disjuncts = []
    exit_due_to_timeout = False
    counter = 0

    while(time.time()-new_start_time<timeout):

        counter += 1
        inner_timer = time.time()

        print('loop number ', counter, 'of the disjuncts cycle.')

        failed_disjuncts, exit_due_to_timeout = mcmh_object.verifyUnverifiedDisjunctsWithMarabou(add_to_goodset=False,
                                                                                                 timeout=int(timeout*0.75),
                                                                                                 verbosity=2)
        if exit_due_to_timeout:
            'time out in verifyUnverifiedDisjunctsWithMarabou'
        print('time verifyUnverifiedDisjunctsWithMarabou took: ', time.time()-inner_timer)

        if exit_due_to_timeout or not failed_disjuncts:
            break

        mcmh_object.adjustDisjunctsOnBadInputs(failed_disjuncts)

        print('failed disjuncts, observed bounds adjusted: ', failed_disjuncts)

        print('Verifying conjunction again.')

        conjunction_time = time.time()

        conjunction_verified, conjunction_counter = repeated_conjunction_verification(mcmh_object)
        if not conjunction_verified:
            print('failure to verify conjunction')
            print('number of loops: ', conjunction_counter)
            break

        print('conjunction modified and verified. Time took', time.time() - conjunction_time,
              '\nNumber of loops: ', conjunction_counter)
        print('Overall time since the beginning of disjunction cycle: ', time.time()-inner_timer)


    print('\n\nLoop done.')
    print('number of loops done:', counter)
    print('Time elapsed since starting the cycle of disjuncts: ', time.time()-new_start_time)

    if exit_due_to_timeout:
        print('time out!')
    elif not failed_disjuncts:
        print('Success! No failed disjuncts.')

    if failed_disjuncts:
        print('failed disjuncts: ', failed_disjuncts)

    print('\n\nVerifying the original network with Marabou!')

    marabou_time = time.time()



    # MarabouCore.createInputQuery(self.ipq2, self.network_filename2, self.property_filename2)

    vals, stats = solve_query(mcmh_object.marabou_nnet.ipq2,verbose=True,verbosity=0,timeout=int(timeout*1.5))

    bad_input = mcmh_object.convertVectorFromDictToList(vals)

    print('Time elapsed since the beginning of marabou verification: ', time.time()-marabou_time)

    if stats.hasTimedOut():
        print('time out!')
    elif bad_input:  # SAT
        print('SAT! Bad input: ', bad_input)
    else:
        print('UNSAT!')


sys.exit(0)





# mcmh_object.layer_interpolant_candidate.excludeFromInvariant(var=14, side='r')

new_start_time = time.time()

conjunction_verified = False
counter = 0
while(time.time()-new_start_time<18000):
    counter += 1
    current_time = time.time()


    print(mcmh_object.layer_interpolant_candidate.getConjunction())
    MarabouCore.createInputQuery(mcmh_object.ipq2, mcmh_object.network_filename2, mcmh_object.property_filename2)
    print(mcmh_object.ipq2.getUpperBound(14))
    print(mcmh_object.ipq2.getLowerBound(14))


    bad_input = mcmh_object.verifyConjunctionWithMarabou(add_to_badset=True)
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

print('new conjunction modified and verified. Time elapsed', time.time() - new_start_time)
sys.exit(0)




sys.exit(0)


