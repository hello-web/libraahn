using System;
using System.Collections.Generic;

namespace Raahn
{
	public class NeuralNetwork
	{
        public delegate double ActivationFunctionType(double x);

        //Default to using the logistic function.
        public ActivationFunctionType activation = Activation.Logistic;
        public ActivationFunctionType activationDerivative = Activation.LogisticDerivative;
        private double learningRate;
        private List<List<NeuronGroup>> allListGroups;
        private List<NeuronGroup> inputGroups;
        private List<NeuronGroup> hiddenGroups;
        private List<NeuronGroup> outputGroups;

		public NeuralNetwork(double lRate)
		{
            learningRate = lRate;

            allListGroups = new List<List<NeuronGroup>>();

            inputGroups = new List<NeuronGroup>();
            hiddenGroups = new List<NeuronGroup>();
            outputGroups = new List<NeuronGroup>();

            allListGroups.Add(inputGroups);
            allListGroups.Add(hiddenGroups);
            allListGroups.Add(outputGroups);
		}

        public void PropagateSignal()
        {
            //Reset the computed state of hidden layer neurons.
            //Also reset the values for hidden and output neurons.
            for (int i = 0; i < hiddenGroups.Count; i++)
            {
                hiddenGroups[i].computed = false;
                hiddenGroups[i].Reset();
            }

            for (int i = 0; i < outputGroups.Count; i++)
                outputGroups[i].Reset();

            for (int i = 0; i < outputGroups.Count; i++)
                outputGroups[i].ComputeSignal();
        }

        public void Train()
        {
            for (int y = 0; y < inputGroups.Count; y++)
                inputGroups[y].Train();

            for (int y = 0; y < hiddenGroups.Count; y++)
                hiddenGroups[y].Train();
        }

        public void DisplayWeights()
        {
            Console.WriteLine("Displaying outgoing weights.\n");

            //Display outgoing connections from input groups.
            for (int x = 0; x < allListGroups[0].Count; x++)
            {
                Console.WriteLine("Displaying weights for input group {0}", x);
                allListGroups[0][x].DisplayOutgoingWeights();
            }

            //Display outgoing connections from hidden groups.
            for (int x = 0; x < allListGroups[1].Count; x++)
            {
                Console.WriteLine("Displaying weights for output group {0}:", x);
                allListGroups[1][x].DisplayOutgoingWeights();
            }
        }

        //Returns false if the input group doesn't exist, or the data is too short. True otherwise.
        //If too many inputs are provided, the excess is discarded.
        public bool SetInputs(uint groupIndex, double[] data)
        {
            if (groupIndex >= inputGroups.Count)
                return false;
            if (data.Length < inputGroups[(int)groupIndex].neurons.Count)
                return false;

            int groupIndexi = (int)groupIndex;

            for (int i = 0; i < inputGroups[groupIndexi].neurons.Count; i++)
                inputGroups[groupIndexi].neurons[i] = data[i];

            return true;
        }

        //Returns false if one or both of the groups do not exist.
        //Returns true if the groups could be connected.
        public bool ConnectGroups(NeuronGroup.Identifier input, NeuronGroup.Identifier output, 
                                  ConnectionGroup.TrainFunctionType trainMethod, uint modulationIndex, bool useBias)
        {
            if (!VerifyIdentifier(input) || !VerifyIdentifier(output))
                return false;

            NeuronGroup iGroup = allListGroups[(int)input.type][(int)input.index];
            NeuronGroup oGroup = allListGroups[(int)output.type][(int)output.index];

            Random rand = new Random();

            ConnectionGroup cGroup = new ConnectionGroup(this, iGroup, oGroup, useBias);
            cGroup.SetTrainingMethod(trainMethod);
            cGroup.SetModulationIndex((int)modulationIndex);

            for (uint x = 0; x < iGroup.neurons.Count; x++)
            {
                //Weights randomized between 0.0 and 1.0.
                for (uint y = 0; y < oGroup.neurons.Count; y++)
                    cGroup.AddConnection(x, y, rand.NextDouble());
            }

            if (useBias)
                cGroup.AddBiasWeights((uint)oGroup.neurons.Count, rand.NextDouble());

            iGroup.AddOutgoingGroup(cGroup);
            oGroup.AddIncomingGroup(cGroup);

            return true;
        }

        //Returns the index of the neuron group.
        public int AddNeuronGroup(uint neuronCount, NeuronGroup.Type type)
        {
            if (!VerifyType(type))
                return NeuronGroup.INVALID_NEURON_INDEX;

            NeuronGroup newGroup = new NeuronGroup(this, type);
            newGroup.AddNeurons(neuronCount);

            switch (type)
            {
                case NeuronGroup.Type.INPUT:
                {
                    inputGroups.Add(newGroup);
                    return inputGroups.Count - 1;
                }
                case NeuronGroup.Type.HIDDEN:
                {
                    hiddenGroups.Add(newGroup);
                    return hiddenGroups.Count - 1;
                }
                case NeuronGroup.Type.OUTPUT:
                {
                    outputGroups.Add(newGroup);
                    return outputGroups.Count - 1;
                }
                default:
                {
                    return NeuronGroup.INVALID_NEURON_INDEX;
                }
            }
        }

        //Returns double.Nan if the neuron or neuron group does not exist.
        public double GetNeuronValue(NeuronGroup.Identifier ident, uint neuronIndex)
        {
            if (!VerifyIdentifier(ident))
                return double.NaN;

            int typei = (int)ident.type;
            int indexi = (int)ident.index;

            if (neuronIndex >= allListGroups[typei][indexi].neurons.Count)
                return double.NaN;

            return allListGroups[typei][indexi].neurons[(int)neuronIndex];
        }

        public double GetLearningRate()
        {
            return learningRate;
        }

        //Makes sure a type is INPUT, HIDDEN, or OUTPUT.
        private bool VerifyType(NeuronGroup.Type type)
        {
            int typei = (int)type;

            if (typei < (int)NeuronGroup.Type.INPUT || typei > (int)NeuronGroup.Type.OUTPUT)
                return false;
            else
                return true;
        }

        //Makes sure an identifier specifies a neuron group within allListGroups.
        private bool VerifyIdentifier(NeuronGroup.Identifier ident)
        {
            if (!VerifyType(ident.type))
                return false;

            if (ident.index < 0 || ident.index >= allListGroups[(int)ident.type].Count)
                return false;

            return true;
        }
	}
}
