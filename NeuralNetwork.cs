using System;
using System.Linq;
using System.Collections.Generic;

namespace Raahn
{
    public partial class NeuralNetwork
    {
        public delegate double ActivationFunctionType(double x);

        private const uint DEFAULT_HISTORY_BUFFER_SIZE = 1;

        public static readonly Random rand = new Random();

        public double noiseMagnitude = 1.0;
        public double noiseLowerBound;
        //Default to using the logistic function.
        public ActivationFunctionType activation = Activation.Logistic;
        public ActivationFunctionType activationDerivative = Activation.LogisticDerivative;
        private uint historyBufferSize;
        private double weightCap;
        private Queue<List<double>> historyBuffer;
        private List<List<NeuronGroup>> allListGroups;
        private List<NeuronGroup> inputGroups;
        private List<NeuronGroup> hiddenGroups;
        private List<NeuronGroup> outputGroups;

        public NeuralNetwork()
        {
            Construct(DEFAULT_HISTORY_BUFFER_SIZE);
        }

        public NeuralNetwork(uint historySize)
        {
            if (historySize > 0)
                Construct(historySize);
            else
                Construct(DEFAULT_HISTORY_BUFFER_SIZE);
        }

        //Adds a training sample to the history buffer.
        public void AddSample(List<double> newSample)
        {
            //If the buffer is full, pop the back and add the new sample.
            if (historyBuffer.Count == historyBufferSize)
                historyBuffer.Dequeue();

            historyBuffer.Enqueue(newSample);

            SetSample(newSample);
        }

        //Propagates the inputs completely and gets output.
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
            for (int i = 0; i < inputGroups.Count; i++)
                inputGroups[i].TrainRecent();

            for (int i = 0; i < inputGroups.Count; i++)
                inputGroups[i].TrainSeveral();

            for (int i = 0; i < hiddenGroups.Count; i++)
                hiddenGroups[i].TrainRecent();

            for (int i = 0; i < hiddenGroups.Count; i++)
                hiddenGroups[i].TrainSeveral();
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

        //Resets the weights and every neuron of the neural network.
        public void Reset()
        {
            for (int x = 0; x < allListGroups.Count; x++)
            {
                for (int y = 0; y < allListGroups[x].Count; y++)
                {
                    allListGroups[x][y].Reset();
                    allListGroups[x][y].ResetOutgoingGroups();
                }
            }
        }

        //Sets the maximum weight value. Ignores sign.
        public void SetWeightCap(double cap)
        {
            weightCap = Math.Abs(cap);
        }

        public void SetOutput(uint groupIndex, uint index, double value)
        {
            outputGroups[(int)groupIndex].neurons[(int)index] = value;
        }

        //Returns false if one or both of the groups do not exist.
        //Returns true if the groups could be connected.
        //Sample count refers to how many training samples should be used each time Train() is called.
        public bool ConnectGroups(NeuronGroup.Identifier input, NeuronGroup.Identifier output, 
                                  ConnectionGroup.TrainFunctionType trainMethod, int modulationIndex, 
                                  uint sampleCount, double learningRate, bool useBias)
        {
            if (!VerifyIdentifier(input) || !VerifyIdentifier(output))
                return false;

            NeuronGroup iGroup = allListGroups[(int)input.type][(int)input.index];
            NeuronGroup oGroup = allListGroups[(int)output.type][(int)output.index];

            ConnectionGroup cGroup = new ConnectionGroup(this, iGroup, oGroup, useBias);
            cGroup.sampleUsageCount = sampleCount;
            cGroup.SetLearningRate(learningRate);
            cGroup.SetTrainingMethod(trainMethod);
            cGroup.SetModulationIndex(modulationIndex);

            for (uint x = 0; x < iGroup.neurons.Count; x++)
            {
                //Weights randomized between 0.0 and 1.0.
                for (uint y = 0; y < oGroup.neurons.Count; y++)
                    cGroup.AddConnection(x, y, rand.NextDouble());
            }

            if (useBias)
                cGroup.AddBiasWeights((uint)oGroup.neurons.Count);

            //sampleCount of zero refers to training off of the most recent experience.
            iGroup.AddOutgoingGroup(cGroup, sampleCount == 0);
            oGroup.AddIncomingGroup(cGroup);

            return true;
        }

        //Gets the number of neurons in a group. Returns 0 if the group is invalid.
        public uint GetGroupNeuronCount(NeuronGroup.Identifier ident)
        {
            if (!VerifyIdentifier(ident))
                return 0;

            return allListGroups[(int)ident.type][ident.index].GetNeuronCount();
        }

        //Returns the index of the neuron group.
        public int AddNeuronGroup(uint neuronCount, NeuronGroup.Type type)
        {
            if (!VerifyType(type))
                return NeuronGroup.INVALID_NEURON_INDEX;

            NeuronGroup newGroup = new NeuronGroup(this, type);
            newGroup.AddNeurons(neuronCount);
            newGroup.type = type;

            switch (type)
            {
                case NeuronGroup.Type.INPUT:
                {
                    inputGroups.Add(newGroup);
                    int groupIndex = inputGroups.Count - 1;

                    newGroup.index = groupIndex;

                    return groupIndex;
                }
                case NeuronGroup.Type.HIDDEN:
                {
                    hiddenGroups.Add(newGroup);
                    int groupIndex = hiddenGroups.Count - 1;

                    newGroup.index = groupIndex;

                    return groupIndex;
                }
                case NeuronGroup.Type.OUTPUT:
                {
                    outputGroups.Add(newGroup);
                    int groupIndex = outputGroups.Count - 1;

                    newGroup.index = groupIndex;

                    return groupIndex;
                }
                default:
                {
                    return NeuronGroup.INVALID_NEURON_INDEX;
                }
            }
        }

        public double GetWeightCap()
        {
            return weightCap;
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

        //Returns double.Nan if the neuron or neuron group does not exist.
        public double GetOutputValue(uint groupIndex, uint index)
        {
            if (groupIndex >= outputGroups.Count)
                return double.NaN;

            if (index >= outputGroups[(int)groupIndex].neurons.Count)
                return double.NaN;

            return outputGroups[(int)groupIndex].neurons[(int)index];
        }

		//Returns the sum of the squared error for the current sample.
		public double GetReconstructionError()
		{
			PropagateSignal();

			double error = 0.0;

			for (int i = 0; i < inputGroups.Count; i++)
				error += inputGroups[i].GetReconstructionError();

			for (int i = 0; i < hiddenGroups.Count; i++)
				error += hiddenGroups[i].GetReconstructionError();

			return error;
		}

		//Returns the average reconstruction error for the entire history buffer.
		public double GetAverageReconstructionError()
		{
			double error = 0.0;

			foreach (List<double> sample in historyBuffer) 
			{
				SetSample(sample);

				error += GetReconstructionError();
			}

			error /= historyBuffer.Count;

			return error;
		}

        //Get neuron values of a neuron group.
        public List<double> GetNeuronValues(NeuronGroup.Identifier nGroup)
        {
            if (!VerifyIdentifier(nGroup))
                return null;

            List<double> neuronValues = new List<double>();

            for (int i = 0; i < allListGroups[(int)nGroup.type][nGroup.index].neurons.Count; i++)
                neuronValues.Add(allListGroups[(int)nGroup.type][nGroup.index].neurons[i]);

            return neuronValues;
        }

        //Get the strength of connections in a connection group.
        public List<double> GetWeights(NeuronGroup.Identifier fromGroup, NeuronGroup.Identifier toGroup)
        {
            if (!VerifyIdentifier(fromGroup) || !VerifyIdentifier(toGroup))
                return null;

            return allListGroups[(int)fromGroup.type][fromGroup.index].GetWeights(toGroup);
        }

        //Returns the Ids of all groups connected by outgoing connections to the specifed group.
        public List<NeuronGroup.Identifier> GetGroupsConnected(NeuronGroup.Identifier connectedTo)
        {
            if (!VerifyIdentifier(connectedTo))
                return null;

            return allListGroups[(int)connectedTo.type][connectedTo.index].GetGroupsConnected();
        }

        private void Construct(uint historySize)
        {
            weightCap = double.MaxValue;

            historyBufferSize = historySize;
            historyBuffer = new Queue<List<double>>((int)historyBufferSize);

            allListGroups = new List<List<NeuronGroup>>();

            inputGroups = new List<NeuronGroup>();
            hiddenGroups = new List<NeuronGroup>();
            outputGroups = new List<NeuronGroup>();

            allListGroups.Add(inputGroups);
            allListGroups.Add(hiddenGroups);
            allListGroups.Add(outputGroups);
        }

        private void SetSample(List<double> sample)
        {
            int dataCount = sample.Count;
            int sampleIndex = 0;

            for (int x = 0; x < inputGroups.Count; x++)
            {
                if (dataCount <= 0)
                    break;

                int neuronCount = inputGroups[x].neurons.Count;

                //If there is to little data use only what is provided.
                if (dataCount < neuronCount)
                    neuronCount = dataCount;

                for (int y = 0; y < neuronCount; y++)
                {
                    inputGroups[x].neurons[y] = sample[sampleIndex];
                    sampleIndex++;
                }

                dataCount -= neuronCount;
            }
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