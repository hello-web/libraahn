using System;
using System.Collections.Generic;

namespace Raahn
{
    public class Connection
    {
        public uint input;
        public uint output;
        public double weight;

        public Connection()
        {
            input = 0;
            output = 0;
            weight = 0.0;
        }

        public Connection(uint i, uint o, double w)
        {
            input = i;
            output = o;
            weight = w;
        }
    }

    public class ConnectionGroup
    {
        public delegate void TrainFunctionType(int modIndex, NeuralNetwork ann, NeuronGroup inGroup,
                                               NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights);

        private int modSigIndex;
        private List<double> biasWeights;
        private List<Connection> connections;
        private NeuralNetwork ann;
        private NeuronGroup inputGroup;
        private NeuronGroup outputGroup;
        private TrainFunctionType trainingMethod;
        //Used for reseting weights.
        private Random rand;

        public ConnectionGroup(NeuralNetwork network, NeuronGroup inGroup, NeuronGroup outGroup, bool useBias)
        {
            connections = new List<Connection>();

            modSigIndex = ModulationSignal.INVALID_INDEX;

            ann = network;

            inputGroup = inGroup;
            outputGroup = outGroup;

            //Default to autoencoder training.
            trainingMethod = TrainingMethod.AutoencoderTrain;

            if (useBias)
                biasWeights = new List<double>();
            else
                biasWeights = null;

            rand = new Random();
        }

        public void AddConnection(uint inputIndex, uint outputIndex, double weight)
        {
            connections.Add(new Connection(inputIndex, outputIndex, weight));
        }

        public void AddBiasWeights(uint outputCount, double weight)
        {
            if (biasWeights == null)
                return;

            for (uint i = 0; i < outputCount; i++)
                biasWeights.Add(weight);
        }

        public void PropagateSignal()
        {
            //Make sure the input group is computed.
            if (!inputGroup.computed)
                inputGroup.ComputeSignal();

            //Use the output neurons to temporarily hold the sum of
            //the weighted inputs. Then apply the activation function.
            for (int i = 0; i < connections.Count; i++)
                outputGroup.neurons[(int)connections[i].output] += inputGroup.neurons[(int)connections[i].input]
                * connections[i].weight;

            if (biasWeights != null)
            {
                for (int i = 0; i < biasWeights.Count; i++)
                    outputGroup.neurons[i] += biasWeights[i];
            }
        }

        public void Train()
        {
            trainingMethod(modSigIndex, ann, inputGroup, outputGroup, connections, biasWeights);
        }

        public void SetModulationIndex(int index)
        {
            modSigIndex = index;
        }

        public void SetTrainingMethod(TrainFunctionType method)
        {
            trainingMethod = method;
        }

        public void DisplayWeights()
        {
            for (int i = 0; i < connections.Count; i++)
                Console.WriteLine(connections[i].weight);

            if (biasWeights != null)
            {
                for (int i = 0; i < biasWeights.Count; i++)
                    Console.WriteLine(biasWeights[i]);
            }
        }

        public void ResetWeights()
        {
            for (int i = 0; i < connections.Count; i++)
                connections[i].weight = rand.NextDouble();
        }

        //Returns true if the connection was able to be removed, false otherwise.
        public bool RemoveConnection(uint index)
        {
            if (index < connections.Count)
            {
                connections.RemoveAt((int)index);
                return true;
            }
            else
                return false;
        }
    }
}
