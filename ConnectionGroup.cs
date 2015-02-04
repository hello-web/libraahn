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

        public ConnectionGroup(NeuralNetwork network, NeuronGroup inGroup, NeuronGroup outGroup, bool useBias)
        {
            connections = new List<Connection>();

            modSigIndex = ModulationSignal.INVALID_INDEX;

            ann = network;

            inputGroup = inGroup;
            outputGroup = outGroup;

            //Default to autoencoder training.
            if (useBias)
            {
                biasWeights = new List<double>();
                trainingMethod = TrainingMethod.BiasAutoencoderTrain;
            }
            else
            {
                biasWeights = null;
                trainingMethod = TrainingMethod.AutoencoderTrain;
            }
        }

        public void AddConnection(uint inputIndex, uint outputIndex, double weight)
        {
            connections.Add(new Connection(inputIndex, outputIndex, weight));
        }

        public void PropagateSignal()
        {
            //Initialize all outputs to zero so they can be used
            //to temporarily hold the sum of the weighted inputs.
            //The activation function is applied afterward.
            for (int i = 0; i < outputGroup.neurons.Count; i++)
                outputGroup.neurons[i] = 0.0;

            if (!inputGroup.computed)
                inputGroup.ComputeSignal();

            for (int i = 0; i < connections.Count; i++)
                outputGroup.neurons[(int)connections[i].output] += inputGroup.neurons[(int)connections[i].input]
                * connections[i].weight;
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
            if (biasWeights == null)
            {
                if (method == TrainingMethod.BiasAutoencoderTrain || method == TrainingMethod.BiasHebbianTrain)
                    return;
            }

            trainingMethod = method;
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
