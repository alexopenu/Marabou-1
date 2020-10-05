#!/usr/bin/python3
#SBATCH -c2
#SBATCH --time=2-0


import os
import site
import sys
import getopt
import datetime

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

from maraboupy.CompositionalVerifier import *

import sys

import time

import numpy as np

LIST_FILE = './list_of_tasks_for_python_sbatch_script'

try:
    with open(LIST_FILE, 'r') as f:
        line = f.readline.strip()

        while (line):
            print('Next line in the instruciton file: ', line)

        start_time = time.time()

        NETWORK = '1_4'
        PROPERTY = '1'
        LAYER = 5
        if REMOTE:
            TIMEOUT = 100000
        else:
            TIMEOUT = 600

        REDIRECT_OUTPUT = True
        GUROBI = 'ON'

        VERIFY_ORIGINAL = False

        if __name__ == "__main__":
            try:
                opts, args = getopt.getopt(line,"hson:t:l:p:",["network=","timout=","layer=","property="])
            except getopt.GetoptError:
                print('Illegal argument in the instruction file: ', line)
                sys.exit(5)
            print(opts)
            print(args)
            for opt, arg in opts:
                if opt == '-h':
                    print('<script name> --network=<network> --timout=<timeout> --layer=<layer> --property=<property>')
                    sys.exit(0)
                elif opt in ('-n', "--network"):
                    NETWORK = arg
                elif opt in ("-t", "--timeout"):
                    TIMEOUT = int(arg)
                elif opt in ("-l", "--layer"):
                    LAYER = int(arg)
                elif opt in ("-p", "--property"):
                    PROPERTY = arg
                elif opt == '-s':
                    REDIRECT_OUTPUT = False
                elif opt == '-o':
                    VERIFY_ORIGINAL = True

        current_date_formatted = datetime.datetime.today().strftime ('%d%m%Y')
        stdout_file = MARABOU_DIR + 'cv_test_output_' + str(current_date_formatted) + '_' + NETWORK + '_prop_' + PROPERTY + \
                      '_layer_' + str(LAYER) + '_gurobi_' + GUROBI

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

        print('Files relevant for this run: ')
        print(stdout_file)
        print(property_filename)
        print(network_filename)
        print(output_property_file)
        print(input_property_file)
        print(network_filename1)
        print(network_filename2)

        mcmh_object = CompositionalVerifier(network_filename=network_filename, property_filename=property_filename,
                                            layer=LAYER)


        # mcmh_object.verify(timeout=TIMEOUT, layer=LAYER,
        #                    network_filename1=network_filename1,
        #                    network_filename2=network_filename2,
        #                    property_filename1=output_property_file,
        #                    property_filename2=input_property_file)
        #

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

        if status == 'raw_conjunction_too_weak':
            print('Interpolant search has failed. A counterexample found between observed layer bounds. Check if SAT?')
            print('Bad input is ', argument_list)

        if status == 'timeout':
            print('Timeout after ', argument_list[0], 'seconds.')
            sys.exit(3)

        line = f.readline.strip()
except:
    print("Something went wrong with reading from the instructions file",
          LIST_FILE)
    sys.exit(1)

print('Done with the master script.')
