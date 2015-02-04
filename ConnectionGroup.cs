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
        public delegate void TrainFunctionType(int modIndex,NeuralNetwork neuralNetwork,NeuronGroup inGroup,
                                                NeuronGroup outGroup,List<Connection> connections);

        private int modSigIndex;
        private List<Connection> connections;
        private NeuralNetwork neuralNetwork;
        private NeuronGroup inputGroup;
        private NeuronGroup outputGroup;
        private TrainFunctionType trainingMethod;

        public ConnectionGroup(NeuralNetwork network, NeuronGroup inGroup, NeuronGroup outGroup)
        {
            connections = new List<Connection>();

            modSigIndex = ModulationSignal.INVALID_INDEX;

            neuralNetwork = network;

            inputGroup = inGroup;
            outputGroup = outGroup;
            //Default to autoencoder training.
            trainingMethod = AutoencoderTrain;
        }

        public static void AutoencoderTrain(int modIndex, NeuralNetwork neuralNetwork, NeuronGroup inGroup, 
                                            NeuronGroup outGroup, List<Connection> connections)
        {
            double learningRate = neuralNetwork.GetLearningRate();

            double[] reconstructions = new double[inGroup.neurons.Count];
            double[] errors = new double[reconstructions.Length];

            //First sum the weighted values into the reconstructions to store them.
            for (int i = 0; i < connections.Count; i++)
                reconstructions[(int)connections[i].input] += outGroup.neurons[(int)connections[i].output]
                * connections[i].weight;

            //Apply the activation function after the weighted values are summed.
            //Also calculate the error of the reconstruction.
            for (int i = 0; i < reconstructions.Length; i++)
            {
                reconstructions[i] = NeuralNetwork.activation(reconstructions[i]);
                errors[i] = inGroup.neurons[i] - reconstructions[i];
            }

            //Update the weights with stochastic gradient descent.
            //Hard code derivative to the derivative of the Logistic function for now.
            for (int i = 0; i < connections.Count; i++)
                connections[i].weight += learningRate * errors[(int)connections[i].input]
                * NeuralNetwork.activationDerivative(outGroup.neurons[(int)connections[i].output])
                * outGroup.neurons[(int)connections[i].output];
        }

        public static void HebbianTrain(int modIndex, NeuralNetwork neuralNetwork, NeuronGroup inGroup, 
                                        NeuronGroup outGroup, List<Connection> connections)
        {
            double learningRate = neuralNetwork.GetLearningRate();
            double modSig = ModulationSignal.GetSignal(modIndex);

            for (int i = 0; i < connections.Count; i++)
            {
                connections[i].weight += modSig * learningRate * inGroup.neurons[(int)connections[i].input]
                * outGroup.neurons[(int)connections[i].output];
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
            trainingMethod(modSigIndex, neuralNetwork, inputGroup, outputGroup, connections);
        }

        public void SetModulationIndex(int index)
        {
            modSigIndex = index;
        }

        public void SetTrainingMethod(TrainFunctionType method)
        {
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
