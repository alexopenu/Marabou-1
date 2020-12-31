#!/usr/bin/python3
#SBATCH -c2
#SBATCH --time=2-0

# import os
import sys
# import getopt
import subprocess


LIST_FILE = './list_of_tasks_for_python_sbatch_script_master'
SCRIPT_NAME = './compositional_verifier_tests_remote.py'

print('Reading the command file ', LIST_FILE)

command_list = []

try:
    with open(LIST_FILE, 'r') as f:
        line = f.readline()
        while (line):
            print('Next line in the file: ', line)
            next_command = line.split()
            if next_command[0][0] == '-': #  No script in the command line,
                next_command.insert(0, SCRIPT_NAME)
            print('Next bash command: ', next_command)
            command_list.append(next_command)

            line = f.readline()
except:
    print("Something went wrong with reading from the file",
          LIST_FILE)
    sys.exit(1)

print('Successfully read the command file.')
print('Executing the sequence of commands.')

for next_command in command_list:
    print('Next bash command: ', next_command)
    subprocess.run(next_command)
    print('Finished with command, ', next_command)

print('Done with the master script.')
