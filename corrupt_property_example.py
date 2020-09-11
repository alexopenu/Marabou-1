#!/usr/bin/python3

import os
import sys
from maraboupy.Marabou import *

if '/cs/' in os.getcwd():
    REMOTE = True
else:
    REMOTE = False

if 'MARABOU_DIR' in os.environ.keys():
    MARABOU_DIR  = os.environ['MARABOU_DIR']
else:
    MARABOU_DIR = '.'


print('Remote run: ', REMOTE)

if REMOTE:
    sys.path.append(MARABOU_DIR)

os.chdir(MARABOU_DIR+'/maraboupy')


NETWORK = '2_7'


network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_" + NETWORK + ".nnet"
bad_property_filename = './bad_property_example.txt'


ipq = MarabouCore.InputQuery()

print(network_filename)
print(bad_property_filename)
status=MarabouCore.createInputQuery(ipq, network_filename, bad_property_filename)
print('Input query created properly: ', status)
for index in range(ipq.getNumInputVariables()):
    var = ipq.inputVariableByIndex(index)
    print(ipq.getLowerBound(var), ' <= x'+str(index), ' <= ', ipq.getUpperBound(var))
for index in range(ipq.getNumOutputVariables()):
    var = ipq.outputVariableByIndex(index)
    print(ipq.getLowerBound(var), ' <= y'+str(index), ' <= ', ipq.getUpperBound(var))
solve_query(ipq)



