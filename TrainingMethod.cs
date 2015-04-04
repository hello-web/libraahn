using System.Collections.Generic;

namespace Raahn
{
    public class TrainingMethod
    {
        private const double BIAS_INPUT = 1.0;
        private const double HEBBIAN_SCALE = 2.0;
        //Since sigmoid returns values between 0,1 half
        //the scale will be the distance in both directions.
        private const double HEBBIAN_OFFSET = HEBBIAN_SCALE / 2.0;

        //Autoencoder training with tied weights.
        public static void AutoencoderTrain(int modIndex, NeuralNetwork ann, NeuronGroup inGroup, 
                                            NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights)
        {
            double learningRate = ann.GetLearningRate();

            int reconstructionCount = inGroup.neurons.Count;

            if (biasWeights != null)
                reconstructionCount++;

            //Plus one for the bias neuron.
            double[] reconstructions = new double[reconstructionCount];
            double[] errors = new double[reconstructionCount];

            //If there is a bias neurons, it's reconstruction and error will be the last value in each.
            int biasRecIndex = reconstructions.Length - 1;

            //First sum the weighted values into the reconstructions to store them.
            for (int i = 0; i < connections.Count; i++)
                reconstructions[(int)connections[i].input] += outGroup.neurons[(int)connections[i].output]
                * connections[i].weight;

            if (biasWeights != null)
            {
                for (int i = 0; i < biasWeights.Count; i++)
                    reconstructions[biasRecIndex] += biasWeights[i];
            }

            //Apply the activation function after the weighted values are summed.
            //Also calculate the error of the reconstruction.
            for (int i = 0; i < inGroup.neurons.Count; i++)
            {
                reconstructions[i] = ann.activation(reconstructions[i]);
                errors[i] = inGroup.neurons[i] - reconstructions[i];
            }

            if (biasWeights != null)
            {
                reconstructions[biasRecIndex] = ann.activation(reconstructions[biasRecIndex]);
                errors[biasRecIndex] = BIAS_INPUT - reconstructions[biasRecIndex];
            }

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
            {
                //Normalize to [-1, 1] to allow for postive and negative deltas without modulation.
                double normalizedInput = inGroup.neurons[(int)connections[i].input] * HEBBIAN_SCALE - HEBBIAN_OFFSET;
                double normalizedOutput = outGroup.neurons[(int)connections[i].output] * HEBBIAN_SCALE - HEBBIAN_OFFSET;

                connections[i].weight += modSig * learningRate * normalizedInput * normalizedOutput;
            }

            if (biasWeights != null)
            {
                //The length of biasWeights should always be equal to the length of outGroup.neurons.
                for (int i = 0; i < biasWeights.Count; i++)
                {
                    double normalizedOutput = outGroup.neurons[(int)connections[i].output] * HEBBIAN_SCALE - HEBBIAN_OFFSET;

                    biasWeights[i] += modSig * learningRate * normalizedOutput;
                }
            }
        }
    }
}

