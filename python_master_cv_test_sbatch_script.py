#!/usr/bin/python3
#SBATCH -c2
#SBATCH --time=2-0



LIST_FILE = './list_of_tasks_for_python_sbatch_script'
SCRIPT_NAME = './compositional_verifier_tests_remote.py'

import os
import sys
import getopt

import subprocess

try:
    with open(LIST_FILE, 'r') as f:
        line = f.readline()
        while (line):
            print('Next line in the file: ', line)
            next_command = line.split()
            next_command.insert(0, SCRIPT_NAME)
            print('Next bash command: ', next_command)
            subprocess.run(next_command)
            print('Finished with command, ', next_command)
            line = f.readline()
except:
    print("Something went wrong with reading from the file",
          LIST_FILE)
    sys.exit(1)

print('Done with the master script.')