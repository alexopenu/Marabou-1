
'''
/* *******************                                                        */
/*! \file PropertyParser.py
 ** \verbatim
 ** Top contributors (to current version):
 ** Alex Usvyatsov
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief
 ** Class that parses a property file into a list of strings representing the individual properties.
 **
 ** [[ Add lengthier description here ]]
 **/
'''



import sys
import re

types_of_eq_properties = ['x', 'y', 'ws', 'm']
'''
    'x'     : input property (mentions only input variables)
    'y'     : output property (mentions only output variables)
    'ws'    : hidden variable property (mentions only hidden variables)
    'm'     : mixed

'''


def parseProperty(property_filename):

    '''
        Parses a property file using regular expressions
        Replaces occurrences of x?? (where ?? are digits) by x[??]
        Replaces occurrences of y?? (where ?? are digits) by y[??]
        (for convenience of evaluating the expression with python's parser)
        Returns:
             two dictionaries: equations amd bounds
                each dictionary has as keys the type (type2) of property: 'x','y','m',(mixed),'ws'
                values are lists of properties of the appropriate type
                    e.g., bounds['x'] is a list of bounds on input variables, where x?? has ben replaced by x[??]

             a list of all properties, given as a list of tuples of strings: (type1,type2,property,index)
                where:
                    type1 is 'e' (equation) or 'b' (bound)
                    type2 is 'x','y','ws', or 'm' (for 'mixed'),
                    property is a line from the property file (unchanged)
                    index is the index of a bound/equation in the appropriate list

    '''

    properties = {'x': [], 'y': [], 'ws': [], 'm': []}

    equations = {'x': [], 'y': [], 'ws': [], 'm': []}
    bounds = {'x': [], 'y': [], 'ws': [], 'm': []}

    reg_input = re.compile(r'[x](\d+)')
    # matches a substring of the form x??? where ? are digits

    reg_output = re.compile(r'[y](\d+)')
    # matches a substring of the form y??? where ? are digits

    reg_ws = re.compile(r'[w][s][_](\d+)[_](\d+)')
    # matches a substring of the form ws_???_??? where ? are digits

    reg_equation = re.compile(r'[+-][xy](\d+) ([+-][xy](\d+) )+(<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
    # matches a string that is a legal equation with input (x??) or output (y??) variables


    reg_bound = re.compile(r'[xy](\d+) (<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    # matches a string which represents a legal bound on an input or an output variable

    reg_ws_bound = re.compile(r'[w][s][_](\d+)[_](\d+) (<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    # matches a string which represents a legal bound on a hidden variable



    #try:
    if True:
        # num_bounds = -1 # running index in self.bounds
        # num_eqs = -1    # running index in self.equiations


        with open(property_filename) as f:
            line = f.readline()
            while(line):

                # Computing type2: input/output/ws/mixed
                matched = False
                if (reg_input.match(line)): # input variables present
                    matched = True
                    type2 = 'x'
                if (reg_output.match(line)): # output variables present
                    type2 = 'm' if matched else 'y'
                    matched = True
                if (reg_ws.match(line)): # hidden variable present
                    type2 = 'ws'



                if reg_equation.match(line): # Equation


                    #replace xi by x[i] and yi by y[i]
                    new_str = line.strip()
                    new_str = reg_input.sub(r"x[\1]", new_str)

                    new_str = reg_output.sub(r"y[\1]", new_str)

                    # Debug
                    print('equation')
                    print(new_str)

                    index = len(equations[type2])

                    equations[type2].append(new_str)

                    type1='e'

                    # num_eqs+=1

                elif reg_bound.match(line): # I/O Bound


                    # replace xi by x[i] and yi by y[i]
                    new_str = line.strip()
                    new_str = reg_input.sub(r"x[\1]", new_str)

                    new_str = reg_output.sub(r"y[\1]", new_str)

                    print('bound: ', new_str) #Debug

                    index = len(bounds[type2])

                    bounds[type2].append(new_str)

                    type1 = 'b'

                    # num_bounds+=1
                else:
                    assert reg_ws_bound.match(line) # At this point the line has to match a legal ws bound

                    index = len(bounds['ws'])
                    bounds['ws'].append(line.strip()) # Storing without change

                    type1 = 'b' # Perhaps at some point better add a new type for ws_bound?


                properties['type2'].append({'type1': type1,'type2': type2,'line': line, 'index': index})


                line = f.readline()
        print('successfully read property file: ', property_filename)
        return equations, bounds, properties

    # except:
    #     print('something went wrong while reading the property file', property_filename)
    #     sys.exit(1)

