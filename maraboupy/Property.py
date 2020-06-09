

'''
/* *******************                                                        */
/*! \file Property.py
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
 ** This class represensts property (expression) of a neural network to be verified
 **
 ** [[ Add lengthier description here ]]
 **/
'''



from PropertyParser import *
import parser

types_of_properties = ['x', 'y', 'ws', 'm']
types_of_io_properties = ['x', 'y', 'm']

'''
    'x'     : input property (mentions only input variables)
    'y'     : output property (mentions only output variables)
    'ws'    : hidden variable property (mentions only hidden variables)
    'm'     : mixed

'''


class Property:
    '''
        Python class that represents a property
        Contains two dictionaries: equations and properties
        Keys in the dictionaries are of the type types_of_properties: 'x','y','ws','m' (mixed)
        Each value is a list of strings

        If executables have been computed, also contains two dictionaries of executable versions of the
        properties above (for efficiency)

        Currently only supports properties that mention input and output variables only

        input variable i is referred to as inputs[i]
        output variable i is referred to as output[i]
        Evaluates the expressions (equations and bounds) with python's parser, by
        turning the string into an executable expression

        NOTE: CURRENTLY DOES NOT SUPPORT VERIFICATION OF BOUNDS ON HIDDEN VARIABLES (TO-DO!)
    '''

    def __init__(self,property_filename,compute_executable_bounds = False, compute_executable_equations = False):
        """
        Creates an object of type Property

        Main argument: property file path

        Computes lists of equations and bounds (stored as strings) that can be parsed and evaluated by python's parser

        NOTE: asserts that all equations and bounds are of the legal form,  only mention input (x) ,
        output (y) , or hidden (ws) variables

        NOTE: changes the names of variables  xi to x[i] and yi to y[i]

        If requested, also computes lists of executable expressions (for efficiency of evaluation later)

        Attributes:
            self.properties_list = []   list of all properties; includes tuples of the form (type1,type2,property)
                                        type1 is 'e' (equation) or 'b' (bound)
                                        type2 is 'x', 'y', 'ws', or 'm' (mixed)
                                        property is the original (unchanged) string representing the property


            self.equations              dictionary (type2: list of equations)
                                            where type2 is 'x','y' or 'm' (mixed) , and each equation is a string
            self.bounds                 dictionary (type2: list of bounds)
                                            where bound is a bound on one variable (string)


            self.exec_equations         dictionary (type2: list of equations) (executable)
            self.exec_bounds            dictionaty (type2: list of bounds) (executable)


        """
        self.equations = dict()
        self.bounds = dict()
        self.exec_equations = dict()
        self.exec_bounds = dict()

        self.properties_list = dict()

        for t in types_of_properties:
            self.exec_equations[t] = []
            self.exec_bounds[t] = []

        # self.eq_indices = {('input',-1),('output',-1),('mixed',-1)}
        # self.bdd_indices = {('input',-1),('output',-1)}

        self.exec_bounds_computed = False
        self.exec_equations_computed = False

        if property_filename == "":
            print('No property file!')
            for t in types_of_properties:
                self.equations[t] = []
                self.bounds[t] = []

                self.properties_list[t] = []
        else:
            self.equations, self.bounds, self.properties_list = parseProperty(property_filename)
        if compute_executable_bounds:
            self.compute_executable_bounds()
        if compute_executable_equations:
            self.compute_executable_equations()



    def get_input_bounds(self):
        return self.bounds['x']

    def get_output_bounds(self):
        return self.bounds['y']

    def get_input_equations(self):
        return self.bounds['x']

    def get_output_equation(self):
        return self.bounds['y']

    def get_exec_input_bounds(self):
        if not self.exec_bounds_computed:
            print('executable bounds not computed')
            return []
        return self.exec_bounds['x']

    def get_exec_output_bounds(self):
        if not self.exec_bounds_computed:
            print('executable bounds not computed')
            return []
        return self.exec_bounds['y']

    def get_exec_input_equations(self):
        if not self.exec_equations_computed:
            print('executable equations not computed')
            return []
        return self.exec_equations['x']

    def get_exec_output_equations(self):
        if not self.exec_equations_computed:
            print('executable equations not computed')
            return []
        return self.exec_equations['y']

    def get_exec_mixed_equations(self):
        if not self.exec_equations_computed:
            print('executable equations not computed')
            return []
        return self.exec_equations['m']

    def get_original_x_properties(self):
        x_list = [p['line'] for p in self.properties_list['x']]
        return x_list

    def get_original_y_properties(self):
        y_list = [p['line'] for p in self.properties_list['y']]
        return y_list

    def get_original_ws_properties(self):
        y_list = [p['line'] for p in self.properties_list['ws']]
        return y_list

    def get_original_mixed_properties(self):
        y_list = [p['line'] for p in self.properties_list['m']]
        return y_list

    def mixed_properties_present(self):
        return len(self.properties_list['m'])

    def ws_properties_present(self):
        return len(self.properties_list['ws'])

    def compute_executable_bounds(self,recompute=False):
        """
        Computes the list of executable bounds for efficiency of evaluation
        NOTE: Does nothing if the list is already non-empty and recompute==False
        """
        if recompute:
            self.exec_bounds = dict()

        if not self.exec_bounds:
            for t in types_of_properties:
                self.exec_bounds[t] = []
                for bound in self.bounds[t]:
                    exec_equation = parser.expr(bound).compile()
                    self.exec_bounds[t].append(exec_equation)
        self.exec_bounds_computed = True


    def compute_executable_equations(self,recompute=False):
        """
        Computes the list of executable equations for efficiency of evaluation
        NOTE: Does nothing if the list is already non-empty and recompute==False
        """
        if recompute:
            self.exec_equations = dict()

        if not self.exec_equations:
            for t in types_of_properties:
                self.exec_equations[t] = []
                for equation in self.equations[t]:
                    exec_equation = parser.expr(equation).compile()
                    self.exec_equations[t].append(exec_equation)
        self.exec_equations_computed = True

    def compute_executables(self,recompute=False):
        self.compute_executable_equations(recompute)
        self.compute_executable_bounds(recompute)


    def verify_equations(self,x,y):
        """
        Returns True iff all the equations hold on the input and the output variables
        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        for t in types_of_io_properties:
            for equation in self.equations[t]:
                exec_equation = parser.expr(equation).compile()
                if not eval(exec_equation):
                    return False

        return True

    def verify_bounds(self,x,y):
        """
        Returns True iff all the bounds hold on the input and the output variables
        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        for t in types_of_io_properties:
            for bound in self.bounds[t]:
                exec_equation = parser.expr(bound).compile()
                if not eval(exec_equation):
                    return False

        return True



    def verify_io_property(self,x,y):
        """
        Returns True iff the property holds on the input and the output variables
        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        return self.verify_bounds(x, y) and self.verify_equations(x, y)


    def verify_equations_exec(self,x,y):
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

        for t in types_of_io_properties:
            for exec_equation in self.exec_equations[t]:
                if not eval(exec_equation):
                    return False

        return True

    def verify_bounds_exec(self,x,y):
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

        for t in types_of_io_properties:
            for exec_equation in self.exec_bounds[t]:
                if not eval(exec_equation):
                    return False

        return True

    def verify_io_property_exec(self,input,output):
        """
        Returns True iff the property holds on the input and the output variables
        Verifies using the precomputed executable list
        Asserts that the list is non-empty

        :param x: list (inputs)
        :param y: list (outputs)
        :return: bool

        NOTE: x and y are lists (or np arrays) and they are used in the evaluation function,
        since they are encoded into the expressions in the lists equation and bounds
        """
        return self.verify_bounds_exec(x=input, y=output) and self.verify_equations_exec(x=input, y=output)



    def verify_specific_io_properties(self, x=[], y=[], input=True, output = True, bdds = True, eqs = True,
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


    def property_dict_getter_by_type(self, type1, use_executables = False):
        '''
        returns the appropriate dictionary
            if type1 == 'e', returns the dictionary of equations
            if type1 == 'b', returns the dictionary of bounds
            if use_executables == True, returns the appropriate dictionary of executables
        :param type1: 'e','b'
        :param use_executables: bool
        :return: dictionary
        '''
        assert type1 in ['e','b']

        if use_executables:
            if type1 == 'b':
                assert self.exec_bounds_computed
            else:
                assert self.exec_equations_computed

            if type1 == 'e':
                return self.exec_equations

            return self.exec_bounds

        if type1 == 'e':
            return self.equations

        return self.bounds


    def get_specific_properties(self, type1, type2, use_executables=False):
        '''
        Returns a list of properties of the appropriate types
            e.g. if type1 == 'b' and type2 == 'x' returns the list of bounds on the x variables
            if use_executables==True returns the appropriate list of executables
        :param type1: 'b','e'
        :param type2: type_of_properties
        :param use_executables: bool (if True, returns list of executables)
        :return: list of strings (if use_executables==False) or executable expressions (if use_executables==True)
        '''
        assert type1 in ['b','e']
        assert type2 in types_of_properties

        return self.property_dict_getter_by_type(type1, use_executables)[type2]


    def verify_specific_io_bounds(self, x=[], y=[], input=False, output = False, use_executables=False):
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

        if use_executables:
            assert self.exec_bounds_computed

        if input:
            assert x
        if output:
            assert y

        bounds_to_verify = []

        if input:
            bounds_to_verify += self.get_specific_properties('b','x', use_executables)

        if output:
            bounds_to_verify += self.get_specific_properties('b','y', use_executables)


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

        if use_executables:
            assert self.exec_equations_computed

        if input:
            assert x
        if output:
            assert y

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



    def verify_specific_properties(self, x=[], y=[], input=False, output = False, bdds = False, eqs = False,
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

        for t in types_of_io_properties:
            if not input and (t == 'x' or t == 'm'):
                continue
            if not output and (t == 'y' or t == 'm'):
                continue

            for p in self.properties_list[t]:
                # if not ((p['type2'] == 'x') or (p['type2'] == 'y')):
                #     continue
                if not eqs and p['type1'] == 'e':
                    continue
                if not bdds and p['type1'] == 'b':
                    continue
                # if not input and p['type2'] == 'x':
                #     continue
                # if not output and p['type2'] == 'y':
                #     continue

                if not self.verify_property_by_index(p['index'], x, y, p['type1'], p['type2'], use_executables):
                    return False

        return True


    def verify_property_by_index(self, index, x, y, type1, type2, use_executables=False):
        '''
        verifies the property of type1,type2 by index in the appropriate list
            e.g., if type1 == 'e' and type2 == 'x', the list is self.equations['x']
            and the property to verify is self.equations['x'][index]
        :param index: int
        :param x:
        :param y:
        :param type1: 'e' or 'b'
        :param type2: type_of_property
        :param use_executables: bool
        :return: bool
        '''
        if type1 == 'e':
            if use_executables:
                assert self.exec_equations_computed
            prop = self.exec_equations[type2][index] if use_executables \
                else parser.expr(self.equations[type2][index]).compile()
        elif type1 == 'b':
            prop = self.exec_bounds[type2][index] if use_executables \
                else parser.expr(self.bounds[type2][index]).compile()
        else: # Currently not supported
            return

        return eval(prop)
