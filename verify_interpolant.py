#!/usr/bin/python3

import os
import site
import sys
import getopt

if '/cs' in os.getcwd():
    REMOTE = True
else:
    REMOTE = False

print(REMOTE)

if REMOTE:
    # os.path.join('/cs/usr/alexus/coding/my_Marabou/Marabou-1')
    # site.addsitedir('/cs/usr/alexus/coding/my_Marabou/Marabou-1')
    sys.path.append('/cs/usr/alexus/coding/my_Marabou/Marabou-1')
    os.chdir('/cs/usr/alexus/coding/my_Marabou/Marabou-1/maraboupy')
else:
    os.chdir('/Users/alexus/coding/my_Marabou/Marabou/maraboupy')

# print(sys.path)

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


NETWORK = '2_5'
PROPERTY = '4'
LAYER = 5
if REMOTE:
    TIMEOUT = 40000
else:
    TIMEOUT = 600

# def main(argv):
if __name__ == "__main__":
   # main(sys.argv[1:])
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hn:t:l:p:",["network=","timout=","layer=","property="])
    except getopt.GetoptError:
        print('verify_interpolant.py --network=<network> --timout=<timeout> --layer=<layer> --property=<property>')
        sys.exit(5)
    for opt, arg in opts:
        if opt == '-h':
            print('verify_interpolant.py --network=<network> --timout=<timeout> --layer=<layer> --property=<property>')
            sys.exit(0)
        elif opt in ('-n', "--network"):
            NETWORK = arg
        elif opt in ("-t", "--timeout"):
            TIMEOUT = int(arg)
        elif opt in ("-l", "--layer"):
            LAYER = int(arg)
        elif opt in ("-p", "--property"):
            PROPERTY = arg

network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_" + NETWORK + ".nnet"
print('\nNetwork: ACASXU_experimental_v2a_' + NETWORK + '.nnet\n')
property_filename = "../resources/properties/acas_property_" + PROPERTY+ ".txt"
print('Property: acas_property_' + PROPERTY + '.txt\n')
print("Interpolant search on layer", LAYER, "\n")

property_filename1 = "../resources/properties/acas_property_1.txt"

network_filename1 = "test/ACASXU_experimental_v2a_"+NETWORK+"_input_.nnet"
network_filename2 = "test/ACASXU_experimental_v2a_"+NETWORK+"_output.nnet"

output_property_file = "output_property_test1_"+NETWORK+".txt"
input_property_file = "input_property_" + "acas" + NETWORK + "_prop" + PROPERTY + "_level" + str(LAYER) + "_test1.txt"

disjunct_network_file = "test/ACASXU_experimental_v2a_"+NETWORK+"_disjunct.nnet"

mcmh_object = CompositionalVerifier(network_filename=network_filename, property_filename=property_filename, layer=LAYER)

mcmh_object.prepareForMarabouCandidateVerification(network_filename1=network_filename1,
                                                   network_filename2=network_filename2,
                                                   property_filename1=output_property_file,
                                                   property_filename2=input_property_file,
                                                   network_filename_disjunct=disjunct_network_file)
# mcmh_object.initiateVerificationProcess(N=5000, compute_loose_offsets='range')

test_split_network(mcmh_object.marabou_nnet, mcmh_object.nnet_object1, mcmh_object.nnet_object2, layer=5)

print('Verifying interpolant by splitting the network.')

start_time = time.time()

mcmh_object.verifyInterpolantFromFile(verbosity=3, proceed_to_the_end=False)

new_start_time = time.time()

print('Time first verification took: ', new_start_time-start_time)

print('Verifying interpolant using the original network and hidden neuron property.')

# mcmh_object.verifyInterpolantFromFile(verbosity=3, split_network=False)

current_time = time.time()

print('Time first verification took: ', new_start_time-start_time)

print('Time second verification took: ', current_time-new_start_time)

# def main(argv):
#     try:
#         opts, args = getopt.getopt(argv,"hn:t:l:p:",["network=","timout=","layer=","property="])
#     except getopt.GetoptError:
#         print('verify_interpolant.py --network=<network> --timout=<timeout> --layer=<layer> --property=<property>')
#         sys.exit(5)
#     for opt, arg in opts:
#         if opt == '-h':
#             print('verify_interpolant.py --network=<network> --timout=<timeout> --layer=<layer> --property=<property>')
#             sys.exit(0)
#         elif opt in ("-n", "--network"):
#             NETWORK = arg
#         elif opt in ("-t", "--timeout"):
#             TIMEOUT = int(arg)
#         elif opt in ("-l", "--layer"):
#             LAYER = int(arg)
#         elif opt in ("-p", "--property"):
#             PROPERTY = arg
