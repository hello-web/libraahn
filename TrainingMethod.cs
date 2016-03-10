using System;
using System.Collections.Generic;

namespace Raahn
{
    public partial class NeuralNetwork
    {
        public class TrainingMethod
        {
            public const double BIAS_INPUT = 1.0;
			public const double ERROR_POWER = 2.0;
            public const double NO_ERROR = 0.0;
            private const double HEBBIAN_SCALE = 2.0;
            //Since sigmoid returns values between 0.1 half
            //the scale will be the distance in both directions.
            private const double HEBBIAN_OFFSET = HEBBIAN_SCALE / 2.0;
			private const double SPARSITY_PARAMETER = 1.0;
			private const double SPARSITY_WEIGHT = 0.1;

            //Autoencoder training with tied weights.
            public static double AutoencoderTrain(int modIndex, double learningRate, NeuralNetwork ann, NeuronGroup inGroup, 
			                                      NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights)
            {
                double weightCap = ann.GetWeightCap();

                int reconstructionCount = inGroup.neurons.Count;
				int featureCount = outGroup.neurons.Count;

                //Plus one for the bias neuron.
				if (biasWeights != null)
					reconstructionCount++;

                double[] reconstructions = new double[reconstructionCount];
                double[] errors = new double[reconstructionCount];
				double[] deltas = new double[reconstructionCount];
				//No need to save backprop errors. Regular error is needed for reporting.
				double[] backPropDeltas = new double[featureCount];

                //If there is a bias neuron, it's reconstruction and error will be the last value in each.
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
				//Do the bias weights separately.
                for (int i = 0; i < inGroup.neurons.Count; i++)
                {
                    reconstructions[i] = ann.activation(reconstructions[i]);
                    errors[i] = inGroup.neurons[i] - reconstructions[i];
					deltas[i] = errors[i] * ann.activationDerivative(reconstructions[i]);
                }

                if (biasWeights != null)
                {
                    reconstructions[biasRecIndex] = ann.activation(reconstructions[biasRecIndex]);
                    errors[biasRecIndex] = BIAS_INPUT - reconstructions[biasRecIndex];
					deltas[biasRecIndex] = errors[biasRecIndex] * ann.activationDerivative(reconstructions[biasRecIndex]);
                }

				//Now that all the errors are calculated, the backpropagated error can be calculated.
				//First compute the dot products of each outgoing weight vector and its respective delta.
				//Go through each connection and add its contribution to its respective dot product.
				for (int i = 0; i < connections.Count; i++) 
				{
					int backPropErrorIndex = (int)connections[i].output;
					int errorIndex = (int)connections[i].input;

					backPropDeltas[backPropErrorIndex] += deltas[errorIndex] * connections[i].weight;
				}

				if (biasWeights != null) 
				{
					for (int i = 0; i < biasWeights.Count; i++)
						backPropDeltas[i] += deltas[biasRecIndex] * biasWeights[i];
				}

				//Now multiply the delta by the derivative of the activation function at the feature neurons.
				for (int i = 0; i < featureCount; i++)
					backPropDeltas[i] *= ann.activationDerivative(outGroup.neurons[i]);

                //Update the weights with stochastic gradient descent.
                for (int i = 0; i < connections.Count; i++)
                {
					int inputIndex = (int)connections[i].input;
					int outputIndex = (int)connections[i].output;

                    double errorWeightDelta = learningRate * deltas[inputIndex] * outGroup.neurons[outputIndex];

					double backPropWeightDelta = learningRate * backPropDeltas[outputIndex] * inGroup.neurons[inputIndex];

					double weightDelta = errorWeightDelta + backPropWeightDelta;

                    if (Math.Abs(connections[i].weight + weightDelta) < weightCap)
                        connections[i].weight += weightDelta;
                }

                if (biasWeights != null)
                {
                    for (int i = 0; i < biasWeights.Count; i++)
                    {
                        double errorWeightDelta = learningRate * deltas[biasRecIndex] * outGroup.neurons[i];

						double backPropWeightDelta = learningRate * backPropDeltas[i];

						double weightDelta = errorWeightDelta + backPropWeightDelta;

                        if (Math.Abs(biasWeights[i] + weightDelta) < weightCap)
                            biasWeights[i] += weightDelta;
                    }
                }

                double sumOfSquaredError = 0.0;

                for (int i = 0; i < reconstructionCount; i++)
                    sumOfSquaredError += Math.Pow(errors[i], ERROR_POWER);

                return (sumOfSquaredError / ERROR_POWER);
            }

			//Sparse autoencoder training with tied weights.
			public static double SparseAutoencoderTrain(int modIndex, double learningRate, NeuralNetwork ann, NeuronGroup inGroup, 
			                                            NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights)
			{
				double weightCap = ann.GetWeightCap();

				int reconstructionCount = inGroup.neurons.Count;
				int featureCount = outGroup.neurons.Count;

				//Plus one for the bias neuron.
				if (biasWeights != null)
					reconstructionCount++;

				double[] reconstructions = new double[reconstructionCount];
				double[] errors = new double[reconstructionCount];
				double[] deltas = new double[reconstructionCount];
				//No need to save backprop errors. Regular error is needed for reporting.
				double[] backPropDeltas = new double[featureCount];

				//If there is a bias neuron, it's reconstruction and error will be the last value in each.
				int biasRecIndex = reconstructions.Length - 1;

				//First sum the weighted values into the reconstructions to store them.
				for (int i = 0; i < connections.Count; i++)
					reconstructions[(int)connections[i].input] += outGroup.neurons[(int)connections[i].output] * connections[i].weight;

				if (biasWeights != null)
				{
					for (int i = 0; i < biasWeights.Count; i++)
						reconstructions[biasRecIndex] += biasWeights[i];
				}

				//Apply the activation function after the weighted values are summed.
				//Also calculate the error of the reconstruction.
				//Do the bias weights separately.
				for (int i = 0; i < inGroup.neurons.Count; i++)
				{
					reconstructions[i] = ann.activation(reconstructions[i]);
					errors[i] = inGroup.neurons[i] - reconstructions[i];
					deltas[i] = errors[i] * ann.activationDerivative(reconstructions[i]);
				}

				if (biasWeights != null)
				{
					reconstructions[biasRecIndex] = ann.activation(reconstructions[biasRecIndex]);
					errors[biasRecIndex] = BIAS_INPUT - reconstructions[biasRecIndex];
					deltas[biasRecIndex] = errors[biasRecIndex] * ann.activationDerivative(reconstructions[biasRecIndex]);
				}

				//Now that all the errors are calculated, the backpropagated error can be calculated.
				//First compute the dot products of each outgoing weight vector and its respective delta.
				//Go through each connection and add its contribution to its respective dot product.
				for (int i = 0; i < connections.Count; i++) 
				{
					int backPropErrorIndex = (int)connections[i].output;
					int errorIndex = (int)connections[i].input;

					backPropDeltas[backPropErrorIndex] += deltas[errorIndex] * connections[i].weight;
				}

				if (biasWeights != null) 
				{
					for (int i = 0; i < biasWeights.Count; i++)
						backPropDeltas[i] += deltas[biasRecIndex] * biasWeights[i];
				}

				//Add the spasity term. Also multiply the delta by the 
				//derivative of the activation function at the feature neurons.
				for (int i = 0; i < featureCount; i++) 
				{
					//Sparsity parameter over average reconstruction.
					double paramOverReconst = SPARSITY_PARAMETER / outGroup.averages[i];
					double oneMinusParamOverOneMinusReconst = (1.0 - SPARSITY_PARAMETER) / (1.0 - outGroup.averages[i]);
					double sparsityTerm = SPARSITY_WEIGHT * (oneMinusParamOverOneMinusReconst - paramOverReconst);

					backPropDeltas[i] *= ann.activationDerivative(outGroup.neurons[i]);
                    backPropDeltas[i] -= sparsityTerm;
				}

				//Update the weights with stochastic gradient descent.
				for (int i = 0; i < connections.Count; i++)
				{
					int inputIndex = (int)connections[i].input;
					int outputIndex = (int)connections[i].output;

					double errorWeightDelta = learningRate * deltas[inputIndex] * outGroup.neurons[outputIndex];

					double backPropWeightDelta = learningRate * backPropDeltas[outputIndex] * inGroup.neurons[inputIndex];

					double weightDelta = errorWeightDelta + backPropWeightDelta;

					if (Math.Abs(connections[i].weight + weightDelta) < weightCap)
						connections[i].weight += weightDelta;
					else
						connections[i].weight = Math.Abs(connections[i].weight) * weightCap;
				}

				if (biasWeights != null)
				{
					for (int i = 0; i < biasWeights.Count; i++)
					{
						double errorWeightDelta = learningRate * deltas[biasRecIndex] * outGroup.neurons[i];

						double backPropWeightDelta = learningRate * backPropDeltas[i];

						double weightDelta = errorWeightDelta + backPropWeightDelta;

						if (Math.Abs(biasWeights[i] + weightDelta) < weightCap)
							biasWeights[i] += weightDelta;
						else
							biasWeights[i] = Math.Abs(biasWeights[i]) * weightCap;
					}
				}

				double sumOfSquaredError = 0.0;

				for (int i = 0; i < reconstructionCount; i++)
					sumOfSquaredError += Math.Pow(errors[i], ERROR_POWER);

				return (sumOfSquaredError / ERROR_POWER);
			}

            //Hebbian learning.
            public static double HebbianTrain(int modIndex, double learningRate, NeuralNetwork ann, NeuronGroup inGroup, 
			                                  NeuronGroup outGroup, List<Connection> connections, List<double> biasWeights)
            {
                double modSig = ModulationSignal.GetSignal(modIndex);

                //If the modulation signal is zero there is no weight change.
                if (modSig == ModulationSignal.NO_MODULATION)
                    return NO_ERROR;

                double weightCap = ann.GetWeightCap();

                for (int i = 0; i < connections.Count; i++)
                {
                    //Normalize to [-1, 1] to allow for positive and negative deltas without modulation.
                    double normalizedInput = inGroup.neurons[(int)connections[i].input];
                    double normalizedOutput = outGroup.neurons[(int)connections[i].output] * HEBBIAN_SCALE - HEBBIAN_OFFSET;
                    double noise = (NeuralNetwork.rand.NextDouble() * ann.weightNoiseRange) - ann.weightNoiseMagnitude;

                    double weightDelta = (modSig * learningRate * normalizedInput * normalizedOutput) + noise;

                    if (Math.Abs(connections[i].weight + weightDelta) < weightCap)
                        connections[i].weight += weightDelta;
                    else
                        connections[i].weight = Math.Sign(connections[i].weight) * weightCap;
                }

                if (biasWeights != null)
                {
                    //The length of biasWeights should always be equal to the length of outGroup.neurons.
                    for (int i = 0; i < biasWeights.Count; i++)
                    {
                        double normalizedOutput = outGroup.neurons[i] * HEBBIAN_SCALE - HEBBIAN_OFFSET;
                        double noise = (NeuralNetwork.rand.NextDouble() * ann.weightNoiseRange) - ann.weightNoiseMagnitude;

                        double weightDelta = (modSig * learningRate * normalizedOutput) + noise;

                        if (Math.Abs(biasWeights[i] + weightDelta) < weightCap)
                            biasWeights[i] += weightDelta;
                        else
                            biasWeights[i] = Math.Sign(biasWeights[i]) * weightCap;
                    }
                }

                return NO_ERROR;
            }
        }
    }
}