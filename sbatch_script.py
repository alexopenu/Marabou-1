#!/usr/bin/python3
#SBATCH -c2
#SBATCH --time=2-0


import os
import sys
import getopt

if '/cs' in os.getcwd():
    REMOTE = True
else:
    REMOTE = False

print(REMOTE)

if REMOTE:
    sys.path.append('/cs/usr/alexus/coding/my_Marabou/Marabou-1')
    os.chdir('/cs/usr/alexus/coding/my_Marabou/Marabou-1/maraboupy')
else:
    os.chdir('/Users/alexus/coding/my_Marabou/Marabou/maraboupy')

print(os.getcwd())

try:
    opts, args = getopt.getopt(sys.argv[1:], "hn:t:l:p:", ["network=", "timout=", "layer=", "property="])
except getopt.GetoptError:
    print('Wrong usage')
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


