using System;
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

            public delegate void TrainFunctionType(int modIndex, double learningRate, NeuralNetwork ann, 
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

                Random rand = new Random();

                for (uint i = 0; i < outputCount; i++)
                    biasWeights.Add(rand.NextDouble());
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
                trainingMethod(modSigIndex, learningRate, ann, inputGroup, outputGroup, connections, biasWeights);
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

            public void ResetWeights()
            {
                for (int i = 0; i < connections.Count; i++)
                    connections[i].weight = rand.NextDouble();
            }

            public int GetInputGroupIndex()
            {
                return inputGroup.index;
            }

            public int GetOutputGroupIndex()
            {
                return outputGroup.index;
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