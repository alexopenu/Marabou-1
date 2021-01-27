'''
Top contributors (to current version):
    - Alex Usvyatsov

This file is part of the Marabou project.
Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.

Property represents a property for Marabou input query
'''

import warnings
import sys

import parser
import re

from maraboupy import MarabouCore

# from MarabouCore import *

TYPES_OF_PROPERTIES_BY_VARIABLE = ['x', 'y', 'h', 'm']
TYPES_OF_IO_PROPERTIES_BY_VARIABLE = ['x', 'y', 'm']
TYPES_OF_VARIABLES = ['x', 'y', 'h']
CLASSES_OF_PROPERTY = ['b', 'e']

'''
    'x'     : input property (mentions only input variables)
    'y'     : output property (mentions only output variables)
    'h'    : hidden variable property (mentions only hidden variables)
    'm'     : mixed property 
    
    'b'     : bound
    'e'     : equation

'''


class Property:
    """
        Python class that represents a Marabou property
        Contains two dictionaries: equations and properties
        Keys in the dictionaries are of the type TYPES_OF_PROPERTIES_BY_VARIABLE: 'x','y','h','m'
        Each value is a list of strings

        If executables have been computed, also contains two dictionaries of executable versions of the
        properties above (for efficiency)

        Currently only supports evaluation of properties that mention input and output variables only

        input variable i is referred to as x[i]
        output variable i is referred to as y[i]
        Evaluates the expressions (equations and bounds) with python's parser, by
        turning the string into an executable expression

        TODO: ADD SUPPORT FOR BOUNDS ON HIDDEN VARIABLES.
        TODO: (desirable?) Add/update bounds and equations to Marabou IPQ
    """

    def __init__(self, property_filename, compute_executable_bounds=True, compute_executable_equations=True,
                 compute_marabou_lists=True):
        """
        Creates an object of type Property

        Main argument: property file path

        Computes lists of equations and bounds (stored as strings) that can be parsed and evaluated by python's parser

        NOTE: asserts that all equations and bounds are of the legal form,  only mention input (x) ,
        output (y) , or hidden (ws) variables

        NOTE: changes the names of variables  xi to x[i] and yi to y[i]

        If requested, also computes lists of executable expressions (for efficiency of evaluation later)

        In addition, contains lists marabou_bounds and marabou_equations
        that contain objects more compatible with the Marabou Input Query class

        Attributes:
            self.properties_list = []   list of all properties; includes tuples of the form (class,type,property)
                                        class is 'e' (equation) or 'b' (bound)
                                        type is TYPES_OF_PROPERTIES_BY_VARIABLE: 'x', 'y', 'ws', or 'm'
                                        property is the original (unchanged) string representing the property


            self.equations              dictionary (type: list of equations)
                                            where type is 'x','y' or 'm' (mixed) , and each equation is a string
            self.bounds                 dictionary (type: list of bounds)
                                            where bound is a bound on one variable (string)


            self.exec_equations         dictionary (type: list of equations) (executable)
            self.exec_bounds            dictionaty (type: list of bounds) (executable)

            self.marabou_bounds         list of bounds compatible with Marabou Input Query
            self.marabou_equations      list of equations compatible with Marabou Input Query
        """
        self.equations = dict()
        self.bounds = dict()
        self.exec_equations = dict()
        self.exec_bounds = dict()

        self.properties_list = dict()

        self.marabou_bounds = []
        self.marabou_equations = []
        self.marabou_property_objects_computed = False

        for t in TYPES_OF_PROPERTIES_BY_VARIABLE:
            self.exec_equations[t] = []
            self.exec_bounds[t] = []

        self.exec_bounds_computed = False
        self.exec_equations_computed = False

        if property_filename == "":
            print('No property file!')
            for t in TYPES_OF_PROPERTIES_BY_VARIABLE:
                self.equations[t] = []
                self.bounds[t] = []

                self.properties_list[t] = []
        else:
            self.readPropertyFile(property_filename)

        if compute_executable_bounds:
            self.compute_executable_bounds()
        if compute_executable_equations:
            self.compute_executable_equations()
        if compute_marabou_lists:
            self.computeMarabouPropertyObjects()

    def get_input_bounds(self):
        """Returns the list of bounds on the input variables

        Returns:
            (list of str)
        """
        return self.bounds['x']

    def get_output_bounds(self):
        """Returns the list of bounds on the output variables

        Returns:
            (list of str)
        """
        return self.bounds['y']

    def get_input_equations(self):
        """Returns the list of equations mentioning only the input variables

        Returns:
            (list of str)
        """
        return self.bounds['x']

    def get_output_equation(self):
        """Returns the list of equations mentioning only the output variables

        Returns:
            (list of str)
        """
        return self.bounds['y']

    def get_exec_input_bounds(self):
        """Returns the list of executable bounds on the input variables

        Returns:
            (list of executable expressions)
        """
        if not self.exec_bounds_computed:
            warnings.warn('Executable bounds have not been computed')
            return []
        return self.exec_bounds['x']

    def get_exec_output_bounds(self):
        """Returns the list of executable bounds on the output variables

        Returns:
            (list of executable expressions)
        """
        if not self.exec_bounds_computed:
            warnings.warn('Executable bounds hev not been computed')
            return []
        return self.exec_bounds['y']

    def get_exec_input_equations(self):
        """Returns the list of executable equations mentioning only the input variables

        Returns:
            (list of executable expressions)
        """
        if not self.exec_equations_computed:
            warnings.warn('Executable equations have not been computed')
            return []
        return self.exec_equations['x']

    def get_exec_output_equations(self):
        """Returns the list of executable equations mentioning only the output variables

        Returns:
            (list of executable expressions)
        """
        if not self.exec_equations_computed:
            warnings.warn('Executable equations have not been computed')
            return []
        return self.exec_equations['y']

    def get_exec_mixed_equations(self):
        """Returns the list of executable equations mentioning mixed types of variables

        Returns:
            (list of executable expressions)
        """
        if not self.exec_equations_computed:
            warnings.warn('Executable equations have not been computed')
            return []
        return self.exec_equations['m']

    def get_original_x_properties(self):
        """Returns the list of input properties as they appear in the property file

        Returns:
            (list of str)
        """
        x_list = [p['line'] for p in self.properties_list['x']]
        return x_list

    def get_original_y_properties(self):
        """Returns the list of output properties as they appear in the property file

        Returns:
            (list of str)
        """
        y_list = [p['line'] for p in self.properties_list['y']]
        return y_list

    def get_original_h_properties(self):
        """Returns the list of properties on hidden neurons, as they appear in the property file

        Returns:
            (list of str)
        """

        y_list = [p['line'] for p in self.properties_list['ws']]
        return y_list

    def get_original_mixed_properties(self):
        """Returns the list of properties mentioning mixed types of variables, as they appear in the property file

        Returns:
            (list of str)
        """

        y_list = [p['line'] for p in self.properties_list['m']]
        return y_list

    def mixed_properties_present(self):
        """Returns True if there are properties that mention more than one type of neuron (e.g., input and output),
        False otherwise

        Returns:
            (bool)
        """
        return (len(self.properties_list['m']) > 0)

    def h_properties_present(self):
        """Returns True if there are properties on hidden neurons, False otherwise

        Returns:
            (bool)
        """
        return (len(self.properties_list['h']) > 0)

    def compute_executable_bounds(self, recompute=False):
        """ Computes the list of executable bounds for later efficiency of evaluation
        
        NOTE: Does nothing if the list is already non-empty and recompute==False
        
         Args:
            recompute (bool): if True, recomputes the expressions even if the list is currently non-empty
        """
        if recompute:
            self.exec_bounds = dict()

        if not self.exec_bounds:
            for t in TYPES_OF_PROPERTIES_BY_VARIABLE:
                self.exec_bounds[t] = []
                for bound in self.bounds[t]:
                    exec_equation = parser.expr(bound).compile()
                    self.exec_bounds[t].append(exec_equation)
        self.exec_bounds_computed = True

    def compute_executable_equations(self, recompute=False):
        """ Computes the list of executable equations for later efficiency of evaluation

        NOTE: Does nothing if the list is already non-empty and recompute==False

         Args:
            recompute (bool): if True, recomputes the expressions even if the list is currently non-empty
        """
        if recompute:
            self.exec_equations = dict()

        if not self.exec_equations:
            for t in TYPES_OF_PROPERTIES_BY_VARIABLE:
                self.exec_equations[t] = []
                for equation in self.equations[t]:
                    exec_equation = parser.expr(equation).compile()
                    self.exec_equations[t].append(exec_equation)
        self.exec_equations_computed = True

    def compute_executables(self, recompute=False):
        """Computes the list of executable properties

         Args:
            recompute (bool): if True, recomputes the expressions even if the lists are currently non-empty
        """
        self.compute_executable_equations(recompute)
        self.compute_executable_bounds(recompute)

    def verify_equations(self, x, y):
        """Returns True iff all the equations hold on the input and the output variables

        Args:
            x (list of float): List of values for the input variables
            y (list of float): List of values for the output variables

        Returns:
            (bool)
        """
        for t in TYPES_OF_IO_PROPERTIES_BY_VARIABLE:
            for equation in self.equations[t]:
                exec_equation = parser.expr(equation).compile()
                if not eval(exec_equation):
                    return False
        """
        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        return True

    def verify_bounds(self, x, y):
        """Returns True iff all the bounds hold on the input and the output variables

        Args:
            x (list of float): List of values for the input variables
            y (list of float): List of values for the output variables

        Returns:
            (bool)
        """
        for t in TYPES_OF_IO_PROPERTIES_BY_VARIABLE:
            for bound in self.bounds[t]:
                exec_equation = parser.expr(bound).compile()
                if not eval(exec_equation):
                    return False
        """
        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        return True

    def verify_io_property(self, x, y):
        """
        Returns True iff the property holds on the input and the output variables
        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        return self.verify_bounds(x, y) and self.verify_equations(x, y)

    def verify_equations_exec(self, x, y):
        """
        Returns True iff all the equations hold on the input and the output variables
        Verifies using the precomputed executable list
        Asserts that executables have been computed

        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        assert self.exec_equations_computed

        for t in TYPES_OF_IO_PROPERTIES_BY_VARIABLE:
            for exec_equation in self.exec_equations[t]:
                if not eval(exec_equation):
                    return False

        return True

    def verify_bounds_exec(self, x, y):
        """
        Returns True iff all the bounds hold on the input and the output variables
        Verifies using the precomputed executable list
        Asserts that executables have been computed

        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        assert self.exec_bounds_computed

        for t in TYPES_OF_IO_PROPERTIES_BY_VARIABLE:
            for exec_equation in self.exec_bounds[t]:
                if not eval(exec_equation):
                    return False

        return True

    def verify_io_property_exec(self, x, y):
        """
        Returns True iff the property holds on the x and the y variables
        Verifies using the precomputed executable list
        Asserts that the list is non-empty

        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        return self.verify_bounds_exec(x=x, y=y) and self.verify_equations_exec(x=x, y=y)

    def verify_specific_io_properties(self, x=[], y=[], input=True, output=True, bdds=True, eqs=True,
                                      use_executables=False):
        '''
        verifies specific properties of the input and the output variables
        :param x: list of floats (inputs)
        :param y: list of floats (outputs)
        :param input: bool (verify input properties?)
        :param output: bool (verify output properties?)
        :param bdds: bool (verify bounds?)
        :param eqs: bool (verify equations?)
        :param use_executables: bool (use the executable versions of the properties?)
        :return: bool (True if all the desirable properties hold)
        '''

        if eqs:
            if not self.verify_specific_io_equations(x, y, input, output, use_executables):
                return False
        if bdds:
            if not self.verify_specific_io_bounds(x, y, input, output, use_executables):
                return False
        return True

    def verify_output_properties(self, y, use_executables=False):
        return self.verify_specific_io_properties(x=[], y=y, input=False, output=True, bdds=True, eqs=True,
                                                  use_executables=use_executables)

    def verify_output_eqautions(self, y, use_executables=False):
        return self.verify_specific_io_equations(x=[], y=y, input=False, output=True, use_executables=use_executables)

    def verify_input_equations(self, x, use_executables=False):
        return self.verify_specific_io_equations(x=x, y=[], input=True, output=False, use_executables=use_executables)

    def property_dict_getter_by_type(self, class_of_property: CLASSES_OF_PROPERTY, use_executables=False):
        '''
        returns the appropriate dictionary
            if class_of_property == 'e', returns the dictionary of equations
            if class_of_property == 'b', returns the dictionary of bounds
            if use_executables == True, returns the appropriate dictionary of executables
        :param class_of_property: 'e','b'
        :param use_executables: bool
        :return: dictionary
        '''
        assert class_of_property in CLASSES_OF_PROPERTY

        if use_executables:
            if class_of_property == 'b':
                assert self.exec_bounds_computed
                return self.exec_bounds
            else:
                assert self.exec_equations_computed
                return self.exec_equations

        if class_of_property == 'e':
            return self.equations

        return self.bounds

    def get_specific_properties(self, class_of_property, type_of_property, use_executables=False):
        '''
        Returns a list of properties of the appropriate types
            e.g. if class_of_property == 'b' and type_of_property == 'x' returns the list of bounds on the x variables
            if use_executables==True returns the appropriate list of executables
        :param class_of_property: 'b','e'
        :param type_of_property: type_of_properties
        :param use_executables: bool (if True, returns list of executables)
        :return: list of strings (if use_executables==False) or executable expressions (if use_executables==True)
        '''
        assert class_of_property in CLASSES_OF_PROPERTY
        assert type_of_property in TYPES_OF_PROPERTIES_BY_VARIABLE

        return self.property_dict_getter_by_type(class_of_property, use_executables)[type_of_property]

    def verify_specific_io_bounds(self, x=[], y=[], input=False, output=False, use_executables=False):
        '''
        verifies bounds on input and output variables
        if input is True, verifies self.bounds['x']
        if output is True, verifies self.bounds['y']
        :param x: list of floats (inputs)
        :param y: list of floats (outputs)
        :param input: bool
        :param output: bool
        :param use_executables: bool
        :return: bool
        '''

        # if use_executables:
        #     assert self.exec_bounds_computed
        #
        # if input:
        #     assert x
        # if output:
        #     assert y

        bounds_to_verify = []

        if input:
            bounds_to_verify += self.get_specific_properties('b', 'x', use_executables)

        if output:
            bounds_to_verify += self.get_specific_properties('b', 'y', use_executables)

        exec_bounds = bounds_to_verify if use_executables \
            else [parser.expr(bound).compile() for bound in bounds_to_verify]

        for exec_equation in exec_bounds:
            if not eval(exec_equation):
                return False

        return True

    def verify_specific_io_equations(self, x=[], y=[], input=False, output=False, use_executables=False):
        '''
        verifies equations on input and output variables
        if input is True, verifies self.equations['x']
        if output is True, verifies self.equations['y']
        if both are True, verifies also self.equations['m'] (the mixed ones)
        :param x: list of floats (inputs)
        :param y: list of floats (outputs)
        :param input: bool
        :param output: bool
        :param use_executables: bool
        :return: bool
        '''

        # if use_executables:
        #     assert self.exec_equations_computed
        #
        # if input:
        #     assert x
        # if output:
        #     assert y

        equations_to_verify = []

        if input:
            equations_to_verify += self.get_specific_properties('e', 'x', use_executables)

        if output:
            equations_to_verify += self.get_specific_properties('e', 'y', use_executables)

            if input:
                equations_to_verify += self.get_specific_properties('e', 'm', use_executables)

        exec_equations = equations_to_verify if use_executables \
            else [parser.expr(eq).compile() for eq in equations_to_verify]

        for exec_equation in exec_equations:
            if not eval(exec_equation):
                return False

        return True

    def verify_specific_properties(self, x=[], y=[], input=False, output=False, bdds=False, eqs=False,
                                   use_executables=False):
        '''
        This method is the most general method for verifying properties of specific type only
        Unlike the other methods, it is based on going through the whole self.properties_list, and not
        getting specific properties directly from the appropriate dictionaries
        Hence it is less efficient, and in general one probably wants to avoid using it.

        :param x:
        :param y:
        :param input: bool (verify input properties?)
        :param output: bool (verify output properties?)
        :param bdds: bool (verify bounds?)
        :param eqs: bool (verify equations?)
        :param use_executables: bool (use executable verions of bounds/equations?)
        :return: bool
        '''
        if input:
            assert x
        if output:
            assert y

        for t in TYPES_OF_IO_PROPERTIES_BY_VARIABLE:
            if not input and (t == 'x' or t == 'm'):
                continue
            if not output and (t == 'y' or t == 'm'):
                continue

            for p in self.properties_list[t]:
                # if not ((p['type_of_property'] == 'x') or (p['type_of_property'] == 'y')):
                #     continue
                if not eqs and p['class_of_property'] == 'e':
                    continue
                if not bdds and p['class_of_property'] == 'b':
                    continue
                # if not input and p['type_of_property'] == 'x':
                #     continue
                # if not output and p['type_of_property'] == 'y':
                #     continue

                if not self.verify_property_by_index(p['index'], x, y, p['class_of_property'], p['type_of_property'],
                                                     use_executables=use_executables):
                    return False

        return True

    def verify_property_by_index(self, index, x, y, class_of_property: CLASSES_OF_PROPERTY,
                                 type_of_property: TYPES_OF_PROPERTIES_BY_VARIABLE, use_executables=False):
        '''
        verifies the property of class class_of_property,type_of_property by index in the appropriate list
            e.g., if class_of_property == 'e' and type_of_property == 'x', the list is self.equations['x']
            and the property to verify is self.equations['x'][index]
        :param index: int
        :param x:
        :param y:
        :param class_of_property: 'e' or 'b'
        :param type_of_property: type_of_property
        :param use_executables: bool
        :return: bool
        '''
        if class_of_property == 'e':
            if use_executables:
                assert self.exec_equations_computed
            prop = self.exec_equations[type_of_property][index] if use_executables \
                else parser.expr(self.equations[type_of_property][index]).compile()
        elif class_of_property == 'b':
            prop = self.exec_bounds[type_of_property][index] if use_executables \
                else parser.expr(self.bounds[type_of_property][index]).compile()
        else:  # Other types are currently not supported. TODO: add support.
            return

        return eval(prop)

    def getMarabouBounds(self):
        if not self.marabou_property_objects_computed:
            warnings.warn('Marabou-compatible bounds and equations have not been computed.')
            return None
        return self.marabou_bounds

    def getMarabouEquations(self):
        if not self.marabou_property_objects_computed:
            warnings.warn('Marabou-compatible bounds and equations have not been computed.')
            return None
        return self.marabou_equations

    def updateMarabouInputQuery(self, ipq: MarabouCore.InputQuery):
        if not self.marabou_property_objects_computed:
            print('Computing Marabou-compatible property objects')
            self.computeMarabouPropertyObjects()
        for (variable, indices, type_of_bound, bound) in self.marabou_bounds:
            if variable == 'x':  # Input variable
                var_index = ipq.inputVariableByIndex(indices)
                if type_of_bound == 'l':
                    pass
                # TODO: complete!

    def readPropertyFile(self, property_filename):

        '''
            Parses a property file using regular expressions
            Replaces occurrences of x?? (where ?? are digits) by x[??]
            Replaces occurrences of y?? (where ?? are digits) by y[??]
            (for convenience of evaluating the expression with python's parser)
            Returns:
                 two dictionaries: equations and bounds
                    each dictionary has as keys the type (TYPES_OF_PROPERTIES_BY_VARIABLE) of property:
                    'x','y','m',(mixed),'h'

                    values are lists of properties of the appropriate type
                        e.g., bounds['x'] is a list of bounds on input variables, where x?? has ben replaced by x[??]
                        so for example, bounds['x'] can look like this: ['x[0] >= 0', 'x[1] <= 0.01']

                 a list of all properties, given as a list of tuples of strings:
                 (class_of_property,type_of_property,property,index)
                    where:
                        class_of_property is 'e' (equation) or 'b' (bound)
                        type_of_property is 'x','y','h', or 'm' (for 'mixed'),
                        property is a line from the property file (unchanged)
                        index is the index of a bound/equation in the appropriate list

        '''

        properties = {'x': [], 'y': [], 'h': [], 'm': []}

        equations = {'x': [], 'y': [], 'h': [], 'm': []}
        bounds = {'x': [], 'y': [], 'h': [], 'm': []}

        reg_input = re.compile(r'[x](\d+)')
        # matches a substring of the form x??? where ? are digits

        reg_output = re.compile(r'[y](\d+)')
        # matches a substring of the form y??? where ? are digits

        reg_hidden = re.compile(r'[h][_](\d+)[_](\d+)')
        # matches a substring of the form ws_???_??? where ? are digits

        reg_equation = re.compile(r'[+-][xy](\d+) ([+-][xy](\d+) )+(<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
        # matches a string that is a legal equation with input (x??) or output (y??) variables

        reg_io_bound = re.compile(r'[xy](\d+) (<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
        # matches a string which represents a legal bound on an input or an output variable

        reg_hidden_bound = re.compile(r'[h][_](\d+)[_](\d+) (<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
        # matches a string which represents a legal bound on a hidden variable

        try:
            with open(property_filename) as f:
                line = f.readline()
                while (line):

                    type_of_property = ''

                    # Computing type_of_property: input/output/ws/mixed
                    if (reg_input.search(line)):  # input variables present
                        type_of_property = 'x'
                    if (reg_output.search(line)):  # output variables present
                        type_of_property = 'm' if type_of_property else 'y'
                    if (reg_hidden.search(line)):  # hidden variable present
                        type_of_property = 'm' if type_of_property else 'h'

                    if not type_of_property:
                        print('An expression of an unknown type found while attempting to parse '
                              'property file ', property_filename)
                        print('expression: ', line)
                        sys.exit(1)

                    if reg_equation.match(line):  # Equation

                        # replace xi by x[i] and yi by y[i]
                        new_str = line.strip()
                        new_str = reg_input.sub(r"x[\1]", new_str)

                        new_str = reg_output.sub(r"y[\1]", new_str)

                        # Debug
                        print('equation')
                        print(new_str)

                        index = len(equations[type_of_property])

                        equations[type_of_property].append(new_str)

                        class_of_property = 'e'

                    elif reg_io_bound.match(line):  # I/O Bound

                        # replace xi by x[i] and yi by y[i]
                        new_str = line.strip()
                        new_str = reg_input.sub(r"x[\1]", new_str)

                        new_str = reg_output.sub(r"y[\1]", new_str)

                        print('bound: ', new_str)  # Debug

                        index = len(bounds[type_of_property])

                        bounds[type_of_property].append(new_str)

                        class_of_property = 'b'

                    else:
                        assert reg_hidden_bound.match(line)  # At this point the line has to match a legal ws bound

                        index = len(bounds['ws'])
                        bounds['ws'].append(line.strip())  # Storing without change

                        class_of_property = 'b'  # Perhaps at some point better add a new type for ws_bound?

                    properties[type_of_property].append({'class_of_property': class_of_property,
                                                         'type_of_property': type_of_property, 'line': line,
                                                         'index': index})

                    line = f.readline()
            print('successfully read property file: ', property_filename)
            self.equations = equations
            self.bounds = bounds
            self.properties_list = properties

        except:
            print('Something went wrong while parsing the property file', property_filename)
            sys.exit(1)

    def computeMarabouPropertyObjects(self, recompute=False):
        if self.marabou_property_objects_computed:
            if recompute:
                self.marabou_property_objects_computed = False
                self.marabou_bounds = []
                self.marabou_equations = []
            else:
                return

        for t in TYPES_OF_PROPERTIES_BY_VARIABLE:
            for p in self.properties_list[t]:
                tokens = re.split('\s+', p['line'].rstrip())
                if p['class_of_property'] == 'b':  # Bound
                    assert len(tokens) == 3
                    assert tokens[1] == '<=' or tokens[1] == '>='
                    if tokens[1] == '<=':
                        type_of_bound = 'r'
                    else:
                        type_of_bound = 'l'
                    variable, indices = tokens[0][0], tokens[0][1:]
                    assert variable == t
                    assert variable in TYPES_OF_VARIABLES
                    if variable[0] == 'h':  # Hidden variable: double index
                        indices = re.split('_', indices)
                        assert len(indices) == 2
                        assert re.match('(\d)+$', indices[0])
                        assert re.match('(\d)+$', indices[1])
                        try:
                            indices = [int(s) for s in indices]
                        except:
                            warnings.warn('Unexpected variable index in a bound. Failed to compute Marabou-compatible'
                                          'property objects.')
                            return
                    else:  # x or y
                        assert re.match('(\d)+$', indices)
                        try:
                            indices = int(indices)
                        except:
                            warnings.warn('Unexpected variable index in a bound. Failed to compute Marabou-compatible'
                                          'property objects.')
                            return

                    try:
                        bound = float(tokens[2])
                    except SyntaxError:
                        warnings.warn('Unexpected bound value. Failed to compute Marabou-compatible'
                                      'property objects.')
                        return

                    self.marabou_bounds.append((variable, indices, type_of_bound, bound))

                    continue

                # We have an equation
                assert len(tokens) >= 3
                try:
                    free_coef = float(tokens[-1])
                except SyntaxError:
                    warnings.warn('Unexpected format for the free coefficient in an equation. '
                                  'Failed to compute Marabou-compatible property objects.')
                    return

                if tokens[-2] == '<=':
                    equation_type = MarabouCore.EquationType.LE
                elif tokens[-2] == '>=':
                    equation_type = MarabouCore.EquationType.GE
                elif tokens[-2] == '=':
                    equation_type = MarabouCore.EquationType.EQ
                else:
                    warnings.warn('Unexpected equation type. '
                                  'Failed to compute Marabou-compatible property objects.')
                    return

                list_of_addends = []
                for addend in tokens[:-2]:
                    addend = re.split('([xyh])', addend)

                    assert len(addend) == 3
                    if addend[0] == '+':
                        coeff = 1
                    elif addend[0] == '-':
                        coeff = -1
                    else:
                        try:
                            coeff = float(addend[0])
                        except SyntaxError:
                            warnings.warn('Unexpected format for a coefficient in an equation. '
                                          'Failed to compute Marabou-compatible property objects.')
                            return

                    variable = addend[1]
                    if variable == 'h':
                        match = re.match(r'_(\d+)_(\d+)$', addend[2])
                        assert match
                        indices = match.groups()
                        try:
                            indices = [int(index) for index in indices]
                        except:
                            warnings.warn('Unexpected format for an index for a hidden variable. '
                                          'Failed to compute Marabou-compatible property objects.')
                            return
                    else:  # x or y
                        try:
                            indices = int(addend[2])
                            assert indices >= 0
                        except:
                            warnings.warn('Unexpected index for a variable. '
                                          'Failed to compute Marabou-compatible property objects.')
                            return

                    list_of_addends.append((coeff, variable, indices))

                equation = {'scalar': free_coef, 'eq_type': equation_type, 'addends': list_of_addends}
                self.marabou_equations.append(equation)

        self.marabou_property_objects_computed = True
