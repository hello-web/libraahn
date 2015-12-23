using System;
using System.IO;
using System.Collections.Generic;

namespace Raahn
{
    public partial class NeuralNetwork
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
            public const double DEFAULT_LEARNING_RATE = 0.1;

            public delegate double TrainFunctionType(int modIndex, double learningRate, NeuralNetwork ann, 
			                                         NeuronGroup inGroup, NeuronGroup outGroup, 
			                                         List<Connection> connections, List<double> biasWeights);

            public uint sampleUsageCount;
            private int modSigIndex;
            //Learning rate for all connections within the group.
            private double learningRate;
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
                learningRate = DEFAULT_LEARNING_RATE;

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

            public void AddBiasWeights(uint outputCount)
            {
                if (biasWeights == null)
                    return;

				double neuronInOutCount = (double)(inputGroup.neurons.Count + outputGroup.neurons.Count + 1);

				for (uint i = 0; i < outputCount; i++)
				{
					double range = Math.Sqrt(WEIGHT_RANGE_SCALE / neuronInOutCount);
					//Keep in the range of [-range, range]
					double weight = (rand.NextDouble() * range * DOUBLE_WEIGHT_RANGE) - range;

					biasWeights.Add(weight);
				}
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

			public void UpdateAverages()
			{
				if (trainingMethod == TrainingMethod.SparseAutoencoderTrain)
					outputGroup.UpdateAverages();
			}

            public void SetModulationIndex(int index)
            {
                modSigIndex = index;
            }

            public void SetLearningRate(double lRate)
            {
                learningRate = lRate;
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

			public void SaveWeights()
            {
                StreamWriter writer = new StreamWriter("weights.txt");

                List<List<double>> weights = new List<List<double>>(inputGroup.neurons.Count);

                for (int x = 0; x < inputGroup.neurons.Count; x++)
                {
                    List<double> newList = new List<double>(outputGroup.neurons.Count);

                    for (int y = 0; y < outputGroup.neurons.Count; y++)
                        newList.Add(0.0);

                    weights.Add(newList);
                }

                for (int i = 0; i < connections.Count; i++)
                    weights[(int)connections[i].input][(int)connections[i].output] = connections[i].weight;

                for (int x = 0; x < inputGroup.neurons.Count; x++)
                {
                    for (int y = 0; y < outputGroup.neurons.Count; y++)
                        writer.WriteLine(weights[x][y]);
                }

                if (biasWeights != null)
                {
                    for (int i = 0; i < biasWeights.Count; i++)
                        writer.WriteLine(biasWeights[i]);
                }

				writer.Close();
            }

            public void ResetWeights()
            {
				double neuronInOutCount = (double)(inputGroup.neurons.Count + outputGroup.neurons.Count);

				if (biasWeights != null)
					neuronInOutCount++;

				for (int i = 0; i < connections.Count; i++)
				{
					double range = Math.Sqrt(WEIGHT_RANGE_SCALE / neuronInOutCount);
					//Keep in the range of [-range, range]
					double weight = (rand.NextDouble() * range * DOUBLE_WEIGHT_RANGE) - range;

					connections[i].weight = weight;
				}

                if (biasWeights != null)
                {
					for (int i = 0; i < biasWeights.Count; i++)
					{
						double range = Math.Sqrt(WEIGHT_RANGE_SCALE / neuronInOutCount);
						//Keep in the range of [-range, range]
						double weight = (rand.NextDouble() * range * DOUBLE_WEIGHT_RANGE) - range;

						biasWeights[i] = weight;
					}
                }
            }

            public int GetInputGroupIndex()
            {
                return inputGroup.index;
            }

            public int GetOutputGroupIndex()
            {
                return outputGroup.index;
            }

            public double Train()
            {
                return trainingMethod(modSigIndex, learningRate, ann, inputGroup, outputGroup, connections, biasWeights);
            }

			public double GetReconstructionError()
			{
				//If a autoencoder training was not used there is no reconstruction error.
				if (trainingMethod == TrainingMethod.HebbianTrain)
					return 0.0;

				int reconstructionCount = inputGroup.neurons.Count;

                //Plus one for the bias neuron.
				if (biasWeights != null)
					reconstructionCount++;

				double[] reconstructions = new double[reconstructionCount];
				double[] errors = new double[reconstructionCount];

				//If there is a bias neuron, it's reconstruction and error will be the last value in each.
				int biasRecIndex = reconstructions.Length - 1;

				//First sum the weighted values into the reconstructions to store them.
				for (int i = 0; i < connections.Count; i++)
					reconstructions[(int)connections[i].input] += outputGroup.neurons[(int)connections[i].output]
					* connections[i].weight;

				if (biasWeights != null)
				{
					for (int i = 0; i < biasWeights.Count; i++)
						reconstructions[biasRecIndex] += biasWeights[i];
				}

				//Apply the activation function after the weighted values are summed.
				//Also calculate the error of the reconstruction.
				//Do the bias weights separately.
				for (int i = 0; i < inputGroup.neurons.Count; i++)
				{
					reconstructions[i] = ann.activation(reconstructions[i]);
					errors[i] = inputGroup.neurons[i] - reconstructions[i];
				}

				if (biasWeights != null)
				{
					reconstructions[biasRecIndex] = ann.activation(reconstructions[biasRecIndex]);
					errors[biasRecIndex] = TrainingMethod.BIAS_INPUT - reconstructions[biasRecIndex];
				}

				double sumOfSquaredError = 0.0;

				for (int i = 0; i < reconstructionCount; i++)
					sumOfSquaredError += Math.Pow(errors[i], TrainingMethod.ERROR_POWER);

				return (sumOfSquaredError / TrainingMethod.ERROR_POWER);
			}

            public TrainFunctionType GetTrainingMethod()
            {
                return trainingMethod;
            }

            public bool UsesBiasWeights()
            {
                if (biasWeights != null)
                    return true;

                return false;
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

            public bool IsConnectedTo(NeuronGroup.Identifier toGroup)
            {
                if (outputGroup.type == toGroup.type && outputGroup.index == toGroup.index)
                    return true;

                return false;
            }

            public NeuronGroup.Type GetInputGroupType()
            {
                return inputGroup.type;
            }

            public NeuronGroup.Type GetOutputGroupType()
            {
                return outputGroup.type;
            }

            //Returns a copy of the weights.
            public List<double> GetWeights()
            {
                List<double> weights = new List<double>();

                for (int i = 0; i < connections.Count; i++)
                    weights.Add(connections[i].weight);

                if (biasWeights != null)
                {
                    for (int i = 0; i < biasWeights.Count; i++)
                        weights.Add(biasWeights[i]);
                }

                return weights;
            }
        }
    }
}
