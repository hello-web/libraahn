using System;
using System.Collections.Generic;

namespace Raahn
{
	public class NeuronGroup
	{
        public enum Type
        {
            INPUT = 0,
            HIDDEN = 1,
            OUTPUT = 2
        }

        public struct Identifier
        {
            public int index;
            public Type type;
        }

        public const int INVALID_NEURON_INDEX = -1;
        private const double NEURON_DEFAULT_VALUE = 0.0;

        public bool computed;
		public List<double> neurons;
        //List of pointers to each connection group connected to this neuron group.
        private List<ConnectionGroup> dendriteGroups;
        //List of outgoing connections instantiated by this group.
        private List<ConnectionGroup> axonGroups;

        public NeuronGroup(NeuralNetwork network, Type t)
        {
            Construct(network, t);
        }

		public NeuronGroup(uint count, NeuralNetwork network, Type t)
		{
            Construct(network, t);

            AddNeurons(count);
		}

        public void Construct(NeuralNetwork network, Type t)
        {
            computed = true;

            neurons = new List<double>();

            dendriteGroups = new List<ConnectionGroup>();
            axonGroups = new List<ConnectionGroup>();
        }

        public void AddNeurons(uint count)
        {
            for (uint i = 0; i < count; i++)
                neurons.Add(NEURON_DEFAULT_VALUE);
        }

        public void AddDendriteGroup(ConnectionGroup dendriteGroup)
        {
            dendriteGroups.Add(dendriteGroup);
        }

        public void AddAxonGroup(ConnectionGroup axonGroup)
        {
            axonGroups.Add(axonGroup);
        }

        public void ComputeSignal()
        {
            for (int i = 0; i < dendriteGroups.Count; i++)
                dendriteGroups[i].PropagateSignal();

            //Finish computing the signal by applying the activation function.
            for (int i = 0; i < neurons.Count; i++)
                neurons[i] = NeuralNetwork.activation(neurons[i]);

            computed = true;
        }

        public void Train()
        {
            for (int i = 0; i < axonGroups.Count; i++)
                axonGroups[i].Train();
        }

        //Returns true if the neuron was able to be removed, false otherwise.
        public bool RemoveNeuron(uint index)
        {
            if (index < neurons.Count)
            {
                neurons.RemoveAt((int)index);
                return true;
            }
            else
                return false;
        }
	}
}
