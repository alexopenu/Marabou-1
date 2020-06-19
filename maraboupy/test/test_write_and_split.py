
from MarabouNetworkNNetQuery import *
from  MarabouNetworkNNet import *

from MarabouNetworkNNetExtentions import *

#import filecmp

from subprocess import call

TOL = 1e-8


property_filename = "../resources/properties/acas_property_4.txt"
network_filename = "../resources/nnet/acasxu/ACASXU_experimental_v2a_1_9.nnet"

layer = 2

def test_split_nnet():

    nnet_object = MarabouNetworkNNet(filename=network_filename)

    print(nnet_object.upperBounds, "\n", nnet_object.lowerBounds)

    nnet_object1, nnet_object2 = splitNNet(marabou_nnet=nnet_object,layer=layer)


    #TESTING THE SPLIT FUNCTIONALITY AND DIFFERENT METHODS OF EVALUATION.

    N = 10

    for i in range(N):
            inputs = createRandomInputsForNetwork(nnet_object)

            layer_output = nnet_object.evaluateNetworkToLayer(inputs, last_layer=layer, normalize_inputs=False, normalize_outputs=False, activate_output_layer=True)
            output1 = nnet_object1.evaluateNetworkToLayer(inputs,last_layer=-1, normalize_inputs=False, normalize_outputs=False, activate_output_layer=True
                                                        )

            assert (layer_output == output1).all()

            true_output = nnet_object.evaluateNetworkToLayer(inputs, last_layer=-1, normalize_inputs=False, normalize_outputs=False)
            output2 = nnet_object2.evaluateNetworkToLayer(layer_output,last_layer=-1, normalize_inputs=False, normalize_outputs=False)
            output2b = nnet_object.evaluateNetworkFromLayer(layer_output,first_layer=layer)
            true_outputb = nnet_object.evaluateNetworkFromLayer(inputs)
            true_outputc = nnet_object.evaluateNetwork(inputs,normalize_inputs=False,normalize_outputs=False)

            assert (true_output == output2).all()
            assert (output2 == output2b).all()
            assert (true_outputb == output2b).all()
            assert (true_outputc == output2b).all()


def test_write_to_file():
    output_filename = "test/ACASXU_experimental_v2a_1_9_output.nnet"

    nnet_object = MarabouNetworkNNetQuery(filename=network_filename, property_filename=property_filename)
    nnet_object.writeNNet(output_filename)

    call(['diff',output_filename,network_filename])

def test_split_and_write():
    output_filename = "test/ACASXU_experimental_v2a_1_9_output.nnet"
    output_filename1 = "test/ACASXU_experimental_v2a_1_9_output1.nnet"
    output_filename2 = "test/ACASXU_experimental_v2a_1_9_output2.nnet"

    nnet_object = MarabouNetworkNNetQuery(filename=network_filename, property_filename=property_filename)
    nnet_object.tightenBounds()
    nnet_object.writeNNet(output_filename)

    nnet_object1, nnet_object2 = splitNNet(marabou_nnet=nnet_object, layer=layer)

    print(nnet_object1.normalize)
    nnet_object1.writeNNet(output_filename1)
    nnet_object2.writeNNet(output_filename2)

    nnet_object_a = MarabouNetworkNNetQuery(filename=output_filename, property_filename=property_filename)
    nnet_object1_a = MarabouNetworkNNetQuery(filename=output_filename1)
    nnet_object2_a = MarabouNetworkNNetQuery(filename=output_filename2)



    # COMPARING RESULTS OF THE NETWORKS CREATED FROM NEW FILES TO THE ORIGINALS ONES.
    N = 10
    for i in range(N):
            inputs = createRandomInputsForNetwork(nnet_object_a)

            layer_output = nnet_object_a.evaluateNetworkToLayer(inputs, last_layer=layer, normalize_inputs=False, normalize_outputs=False, activate_output_layer=True)
            output1 = nnet_object1_a.evaluateNetworkToLayer(inputs,last_layer=-1, normalize_inputs=False, normalize_outputs=False, activate_output_layer=True
                                                        )

            if not (layer_output == output1).all():
                   print("Failed1")

            true_output = nnet_object_a.evaluateNetworkToLayer(inputs, last_layer=-1, normalize_inputs=False, normalize_outputs=False)
            output2 = nnet_object2_a.evaluateNetworkToLayer(layer_output,last_layer=-1, normalize_inputs=False, normalize_outputs=False)
            output2b = nnet_object_a.evaluateNetworkFromLayer(layer_output,first_layer=layer)
            true_outputb = nnet_object_a.evaluateNetworkFromLayer(inputs)
            true_outputc = nnet_object_a.evaluateNetwork(inputs,normalize_inputs=False,normalize_outputs=False)
            true_outputd = nnet_object.evaluateNetwork(inputs,normalize_inputs=False,normalize_outputs=False)

            #Test evaluateWithoutMarabou from MarabouNetwork.py
            true_outputf = nnet_object.evaluate(np.array([inputs]),useMarabou=False).flatten().tolist()

            #Test evaluateWithMarabou from MarabouNetwork.py
            true_outpute = nnet_object.evaluate(np.array([inputs])).flatten()
            true_outpute_rounded = np.array([float(round(y,8)) for y in true_outpute])

            if not (true_outputb == output2b).all():
                   print("Failed2")
            if not (true_outputf == output2b).all():
                   print("Failed3")

            # Some inputs lead to different outputs (even though they look the same), when evaluating
            # with and without Marabou

            if not (true_outpute == true_outputc).all():
                   print(true_outpute == true_outputc)
                   print("i=", i, "   input: ", inputs, "   ", "\n", "WithoutMarabou output: ", true_outputc, "\n",
                         "WithMarabou output: ", true_outpute, "\n", "direct output: ", true_outputb, "\n")

            # However, if both are rounded to the same number of digits (say, 8), the results agree:

            true_outputc_rounded = np.array([float(round(y, 8)) for y in true_outputc])

            if not (true_outputc_rounded == true_outpute_rounded).all():
                   print("Failed4")

if __name__ == "__main__":
    test_split_nnet()
    test_write_to_file()
    test_split_and_write()