using System.Collections.Generic;

namespace Raahn
{
    public class TrainingMethod
    {
        private const double BIAS_INPUT = 1.0;

        //Autoencoder training with tied weights.
        public static void AutoencoderTrain(int modIndex, NeuralNetwork ann, NeuronGroup inGroup, 
                                            NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights)
        {
            double learningRate = ann.GetLearningRate();

            //Plus one for the bias neuron.
            double[] reconstructions = new double[inGroup.neurons.Count + 1];
            double[] errors = new double[reconstructions.Length + 1];

            int biasRecIndex = reconstructions.Length - 1;

            //First sum the weighted values into the reconstructions to store them.
            for (int i = 0; i < connections.Count; i++)
                reconstructions[(int)connections[i].input] += outGroup.neurons[(int)connections[i].output]
                * connections[i].weight;

            for (int i = 0; i < biasWeights.Count; i++)
                reconstructions[biasRecIndex] += biasWeights[i];

            //Apply the activation function after the weighted values are summed.
            //Also calculate the error of the reconstruction.
            for (int i = 0; i < inGroup.neurons.Count; i++)
            {
                reconstructions[i] = ann.activation(reconstructions[i]);
                errors[i] = inGroup.neurons[i] - reconstructions[i];
            }

            reconstructions[biasRecIndex] = ann.activation(reconstructions[biasRecIndex]);
            errors[biasRecIndex] = BIAS_INPUT - reconstructions[biasRecIndex];

            //Update the weights with stochastic gradient descent.
            for (int i = 0; i < connections.Count; i++)
                connections[i].weight += learningRate * errors[(int)connections[i].input]
                * ann.activationDerivative(outGroup.neurons[(int)connections[i].output])
                * outGroup.neurons[(int)connections[i].output];

            if (biasWeights != null)
            {
                for (int i = 0; i < biasWeights.Count; i++)
                    biasWeights[i] += learningRate * errors[biasRecIndex] * ann.activationDerivative(outGroup.neurons[i])
                    * outGroup.neurons[i];
            }
        }

        //Hebbian learning.
        public static void HebbianTrain(int modIndex, NeuralNetwork ann, NeuronGroup inGroup, 
                                            NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights)
        {
            double learningRate = ann.GetLearningRate();
            double modSig = ModulationSignal.GetSignal(modIndex);

            for (int i = 0; i < connections.Count; i++)
                connections[i].weight += modSig * learningRate * inGroup.neurons[(int)connections[i].input]
                * outGroup.neurons[(int)connections[i].output];

            if (biasWeights != null)
            {
                //The length of biasWeights should always be equal to the length of outGroup.neurons.
                for (int i = 0; i < biasWeights.Count; i++)
                    biasWeights[i] += modSig * learningRate * outGroup.neurons[i];
            }
        }
    }
}

