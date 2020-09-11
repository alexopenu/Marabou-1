#!/usr/bin/python3

import os
import sys


if '/cs/' in os.getcwd():
    REMOTE = True
    if 'MARABOU_DIR' in os.environ.keys():
        MARABOU_DIR  = os.environ['MARABOU_DIR']
    else:
        MARABOU_DIR = './'
else:
    REMOTE = False

print('Remote run: ', REMOTE)

if REMOTE:
    sys.path.append(MARABOU_DIR)
    os.chdir(MARABOU_DIR+'/maraboupy')

BAD_PROPERTY_FILENAME = './bad_property_example.txt'
NETWORK_FILENAME = './ACASXU_experimental_v2a_2_7_input_.nnet'


