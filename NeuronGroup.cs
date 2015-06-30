using System;
using System.Collections.Generic;

namespace Raahn
{
	public class NeuronGroup
	{
        public enum Type
        {
            NONE = -1,
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

        public int index;
        public bool computed;
        public NeuronGroup.Type type;
		public List<double> neurons;
        private List<ConnectionGroup> incomingGroups;
        private List<ConnectionGroup> outgoingGroups;
        private NeuralNetwork ann;

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
            ann = network;

            computed = true;

            neurons = new List<double>();

            incomingGroups = new List<ConnectionGroup>();
            outgoingGroups = new List<ConnectionGroup>();
        }

        public void AddNeurons(uint count)
        {
            for (uint i = 0; i < count; i++)
                neurons.Add(NEURON_DEFAULT_VALUE);
        }

        public void AddIncomingGroup(ConnectionGroup incomingGroup)
        {
            incomingGroups.Add(incomingGroup);
        }

        public void AddOutgoingGroup(ConnectionGroup outgoingGroup)
        {
            outgoingGroups.Add(outgoingGroup);
        }

        public void Reset()
        {
            for (int i = 0; i < neurons.Count; i++)
                neurons[i] = 0.0;
        }

        public void ResetOutgoingGroups()
        {
            //Weights randomized between 0.0 and 1.0.
            for (int i = 0; i < outgoingGroups.Count; i++)
                outgoingGroups[i].ResetWeights();
        }

        public void ComputeSignal()
        {
            for (int i = 0; i < incomingGroups.Count; i++)
                incomingGroups[i].PropagateSignal();

            //Finish computing the signal by applying the activation function.
            for (int i = 0; i < neurons.Count; i++)
                neurons[i] = ann.activation(neurons[i]);

            computed = true;
        }

        public void Train()
        {
            for (int i = 0; i < outgoingGroups.Count; i++)
                outgoingGroups[i].Train();
        }

        public void DisplayIncomingWeights()
        {
            for (int i = 0; i < incomingGroups.Count; i++)
                incomingGroups[i].DisplayWeights();
        }

        public void DisplayOutgoingWeights()
        {
            for (int i = 0; i < outgoingGroups.Count; i++)
                outgoingGroups[i].DisplayWeights();
        }

        public uint GetNeuronCount()
        {
            uint count = (uint)neurons.Count;

            for (int i = 0; i < outgoingGroups.Count; i++)
            {
                if (outgoingGroups[i].UsesBiasWeights())
                    return count + 1;
            }

            return count;
        }

        public List<double> GetWeights(NeuronGroup.Identifier toGroup)
        {
            for (int i = 0; i < outgoingGroups.Count; i++)
            {
                if (outgoingGroups[i].IsConnectedTo(toGroup))
                    return outgoingGroups[i].GetWeights();
            }

            return null;
        }

        public List<NeuronGroup.Identifier> GetGroupsConnected()
        {
            List<NeuronGroup.Identifier> groupsConnected = new List<NeuronGroup.Identifier>(outgoingGroups.Count);

            for (int i = 0; i < outgoingGroups.Count; i++)
            {
                NeuronGroup.Identifier ident;
                ident.type = outgoingGroups[i].GetOutputGroupType();
                ident.index = outgoingGroups[i].GetOutputGroupIndex();

                if (!groupsConnected.Contains(ident))
                    groupsConnected.Add(ident);
            }

            return groupsConnected;
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
