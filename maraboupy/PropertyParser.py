
import re

class PropertyParser:

    reg_input = re.compile(r'[x](\d+)')
    # matches a substring of the form x??? where ? are digits

    reg_output = re.compile(r'[y](\d+)')
    # matches a substring of the form y??? where ? are digits

    reg_hidden = re.compile(r'[h][_](\d+)[_](\d+)')
    # matches a substring of the form h_???_??? where ? are digits

    reg_equation = re.compile(r'[+-][xy](\d+) ([+-][xy](\d+) )+(<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
    # matches a string that is a legal equation with input (x??) or output (y??) variables

    reg_io_bound = re.compile(r'[xy](\d+) (<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    # matches a string which represents a legal bound on an input or an output variable

    reg_hidden_bound = re.compile(r'[h][_](\d+)[_](\d+) (<=|>=|=) [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    # matches a string which represents a legal bound on a hidden variable



    def parseSingleProperty(line: str):
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
