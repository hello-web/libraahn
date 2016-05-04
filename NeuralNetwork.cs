using System;
using System.Linq;
using System.Collections.Generic;

namespace Raahn
{
    public partial class NeuralNetwork
    {
        //Description of a distance between two novelty buffer occupants.
        private class DistanceDescription : IComparable<DistanceDescription>
        {
            public double distance;
            public NoveltyBufferOccupant distanceOwner;

            public DistanceDescription()
            {
                distance = 0.0;
                distanceOwner = null;
            }

            //Sort DistanceDescription by distance.
            public int CompareTo(DistanceDescription cmp)
            {
                if (cmp == null)
                    return 1;
                if (distance > cmp.distance)
                    return 1;
                if (distance < cmp.distance)
                    return -1;

                return 0;
            }
        }

        private class NoveltyBufferOccupant : IComparable<NoveltyBufferOccupant>
        {
            public double noveltyScore;
            public List<double> experience;
            //Ordered from nearest to farthest.
            public LinkedList<DistanceDescription> distanceDescriptions;

            public NoveltyBufferOccupant()
            {
                noveltyScore = 0.0;

                experience = null;
                distanceDescriptions = new LinkedList<DistanceDescription>();
            }

            //Sort NoveltyBufferOccupant by noveltyScore.
            public int CompareTo(NoveltyBufferOccupant cmp)
            {
                if (cmp == null)
                    return 1;
                if (noveltyScore > cmp.noveltyScore)
                    return 1;
                if (noveltyScore < cmp.noveltyScore)
                    return -1;

                return 0;
            }
        }

        public delegate double ActivationFunctionType(double x);

        public const uint DEFAULT_HISTORY_BUFFER_SIZE = 1;
        //Whether or not to use a novelty buffer.
        private const bool DEFAULT_NOVELTY_USE = false;
        //Number of nearest neighbors to use for novelty score calculations.
        private const uint N_NEAREST = 20;
        //How many distances to keep for each experience.
        private const uint N_KEEP = N_NEAREST + 20;
        private const double DEFAULT_NOISE_MAGNITUDE = 1.0;
        private const double DOUBLE_MAGNITUDE = 2.0;
        private const double WEIGHT_RANGE_SCALE = 6.0;
        private const double DOUBLE_WEIGHT_RANGE = 2.0;

        public static readonly Random rand = new Random();

        public bool useNovelty;
        //Default to using the logistic function.
        public ActivationFunctionType activation = Activation.Logistic;
        public ActivationFunctionType activationDerivative = Activation.LogisticDerivative;
        private uint historyBufferSize;
        private double weightCap;
        private double outputNoiseMagnitude;
        private double weightNoiseMagnitude;
        //Difference between max and min noise values.
        private double outputNoiseRange;
        private double weightNoiseRange;
        private double averageError;
        private List<List<NeuronGroup>> allListGroups;
        private List<NeuronGroup> inputGroups;
        private List<NeuronGroup> hiddenGroups;
        private List<NeuronGroup> outputGroups;
        //Ordered from least novel to most novel.
        private List<NoveltyBufferOccupant> noveltyBuffer;
        private Queue<double> errorBuffer;
        private Queue<List<double>> historyBuffer;

        public NeuralNetwork()
        {
            Construct(DEFAULT_HISTORY_BUFFER_SIZE, DEFAULT_NOISE_MAGNITUDE, DEFAULT_NOISE_MAGNITUDE, DEFAULT_NOVELTY_USE);
        }

        public NeuralNetwork(uint historySize, bool useNoveltyBuffer)
        {
            if (historySize > 0)
                Construct(historySize, DEFAULT_NOISE_MAGNITUDE, DEFAULT_NOISE_MAGNITUDE, useNoveltyBuffer);
            else
                Construct(DEFAULT_HISTORY_BUFFER_SIZE, DEFAULT_NOISE_MAGNITUDE, DEFAULT_NOISE_MAGNITUDE, useNoveltyBuffer);
        }

        public NeuralNetwork(double outputNoiseMag, double weightNoiseMag)
        {
            Construct(DEFAULT_HISTORY_BUFFER_SIZE, outputNoiseMag, weightNoiseMag, DEFAULT_NOVELTY_USE);
        }

        public NeuralNetwork(uint historySize, double outputNoiseMag, double weightNoiseMag, bool useNoveltyBuffer)
        {
            if (historySize > 0)
                Construct(historySize, outputNoiseMag, weightNoiseMag, useNoveltyBuffer);
            else
                Construct(DEFAULT_HISTORY_BUFFER_SIZE, outputNoiseMag, weightNoiseMag, useNoveltyBuffer);
        }

        //Adds a training sample to the history buffer.
        public void AddExperience(List<double> newExperience)
        {
            //Check which buffer to use.
            if (useNovelty)
            {
                //Allocate a NoveltyBufferOccupant for the new experience.
                NoveltyBufferOccupant newOccupant = new NoveltyBufferOccupant();
                newOccupant.experience = newExperience;
                List<DistanceDescription> newDistances = ComputeNewDistances(newOccupant);

                //Check if the buffer is at capacity.
                if (noveltyBuffer.Count == historyBufferSize)
                {
                    NoveltyBufferOccupant leastNovel = noveltyBuffer[0];

                    //If the new experience is more novel than the least novel
                    //then remove the least novel than add the new experience.
                    if (newOccupant.noveltyScore > leastNovel.noveltyScore)
                    {
                        RemoveNoveltyOccupant(leastNovel);
                        //Add the new experience taking into account its newly introduced distances.
                        AddNoveltyOccupant(newOccupant, newDistances);
                    }
                }
                //Buffer not at capacity, just add the new experience.
                else
                    AddNoveltyOccupant(newOccupant, newDistances);
            }
            else
            {
                //If the buffer is full, pop the back to make room for the new experience.
                if (historyBuffer.Count == historyBufferSize)
                    historyBuffer.Dequeue();

                //Add the new experience.
                historyBuffer.Enqueue(newExperience);
            }

            SetExperience(newExperience);
        }

        //Public access for setting an experience.
        public void SetExperience(uint index)
        {
            //Check which buffer to use.
            if (useNovelty)
            {
                //Make sure the index is in the bounds of the novelty buffer.
                if (index >= noveltyBuffer.Count)
                    return;

                SetExperience(noveltyBuffer[(int)index].experience);
            }
            else
            {
                //Make sure the index is in the bounds of the history buffer.
                if (index >= historyBuffer.Count)
                    return;

                SetExperience(historyBuffer.ElementAt((int)index));
            }
        }

        //Propagates the inputs completely and gets output.
        public void PropagateSignal()
        {
            //Reset the computed state of hidden layer neurons.
            //Also reset the values for hidden and output neurons.
            for (int i = 0; i < hiddenGroups.Count; i++)
            {
                hiddenGroups[i].computed = false;
                hiddenGroups[i].Reset();
            }

            for (int i = 0; i < outputGroups.Count; i++)
                outputGroups[i].Reset();

            for (int i = 0; i < outputGroups.Count; i++)
                outputGroups[i].ComputeSignal();
        }

        //Returns autoencoder error.
        public void Train()
        {
            double error = 0.0;

            for (int i = 0; i < inputGroups.Count; i++)
                error += inputGroups[i].TrainRecent();

            for (int i = 0; i < hiddenGroups.Count; i++)
                error += hiddenGroups[i].TrainRecent();

            for (int i = 0; i < inputGroups.Count; i++)
                error += inputGroups[i].TrainSeveral();

            for (int i = 0; i < hiddenGroups.Count; i++)
                error += hiddenGroups[i].TrainSeveral();

            UpdateOnlineError(error);
        }

        //Resets the weights and every neuron of the neural network.
        public void Reset()
        {
            for (int x = 0; x < allListGroups.Count; x++)
            {
                for (int y = 0; y < allListGroups[x].Count; y++)
                {
                    allListGroups[x][y].Reset();
                    allListGroups[x][y].ResetOutgoingGroups();
                }
            }

            if (useNovelty)
                noveltyBuffer.Clear();
            else
                historyBuffer.Clear();

            errorBuffer.Clear();

            averageError = 0.0;
        }

        //Sets the maximum weight value. Ignores sign.
        public void SetWeightCap(double cap)
        {
            weightCap = Math.Abs(cap);
        }

        public void SetOutputNoiseMagnitude(double outputNoiseMag)
        {
            outputNoiseMagnitude = outputNoiseMag;
            outputNoiseRange = outputNoiseMag * DOUBLE_MAGNITUDE;
        }

        public void SetWeightNoiseMagnitude(double weightNoiseMag)
        {
            weightNoiseMagnitude = weightNoiseMag;
            weightNoiseRange = weightNoiseMag * DOUBLE_MAGNITUDE;
        }

        //Returns whether the output was able to be set.
        public bool SetOutput(uint groupIndex, uint index, double value)
        {
            if (groupIndex >= outputGroups.Count)
                return false;

            int groupIndexi = (int)groupIndex;

            if (index >= outputGroups[groupIndexi].neurons.Count)
                return false;

            outputGroups[groupIndexi].neurons[(int)index] = value;

            return true;
        }

        //Returns false if one or both of the groups do not exist.
        //Returns true if the groups could be connected.
        //Sample count refers to how many training samples should be used each time Train() is called.
        public bool ConnectGroups(NeuronGroup.Identifier input, NeuronGroup.Identifier output, 
                                  ConnectionGroup.TrainFunctionType trainMethod, int modulationIndex, 
                                  uint sampleCount, double learningRate, bool useBias)
        {
            if (!VerifyIdentifier(input) || !VerifyIdentifier(output))
                return false;

            NeuronGroup iGroup = allListGroups[(int)input.type][(int)input.index];
            NeuronGroup oGroup = allListGroups[(int)output.type][(int)output.index];

            ConnectionGroup cGroup = new ConnectionGroup(this, iGroup, oGroup, useBias);
            cGroup.sampleUsageCount = sampleCount;
            cGroup.SetLearningRate(learningRate);
            cGroup.SetTrainingMethod(trainMethod);
            cGroup.SetModulationIndex(modulationIndex);

            double neuronInOutCount = (double)(iGroup.neurons.Count + oGroup.neurons.Count);

            if (useBias)
                neuronInOutCount++;

            for (uint x = 0; x < iGroup.neurons.Count; x++)
            {
                //Randomize weights.
                for (uint y = 0; y < oGroup.neurons.Count; y++)
                {
                    double range = Math.Sqrt(WEIGHT_RANGE_SCALE / neuronInOutCount);
                    //Keep in the range of [-range, range]
                    double weight = (rand.NextDouble() * range * DOUBLE_WEIGHT_RANGE) - range;

                    cGroup.AddConnection(x, y, weight);
                }
            }

            if (useBias)
                cGroup.AddBiasWeights((uint)oGroup.neurons.Count);

            //sampleCount of zero refers to training off of the most recent experience.
            iGroup.AddOutgoingGroup(cGroup, sampleCount == 0);
            oGroup.AddIncomingGroup(cGroup);

            return true;
        }

        //Gets the number of neurons in a group. Returns 0 if the group is invalid.
        public uint GetGroupNeuronCount(NeuronGroup.Identifier ident)
        {
            if (!VerifyIdentifier(ident))
                return 0;

            return allListGroups[(int)ident.type][ident.index].GetNeuronCount();
        }

        //Returns the index of the neuron group.
        public int AddNeuronGroup(uint neuronCount, NeuronGroup.Type type)
        {
            if (!VerifyType(type))
                return NeuronGroup.INVALID_NEURON_INDEX;

            NeuronGroup newGroup = new NeuronGroup(this, type);
            newGroup.AddNeurons(neuronCount);
            newGroup.type = type;

            switch (type)
            {
                case NeuronGroup.Type.INPUT:
                {
                    inputGroups.Add(newGroup);
                    int groupIndex = inputGroups.Count - 1;

                    newGroup.index = groupIndex;

                    return groupIndex;
                }
                case NeuronGroup.Type.HIDDEN:
                {
                    hiddenGroups.Add(newGroup);
                    int groupIndex = hiddenGroups.Count - 1;

                    newGroup.index = groupIndex;

                    return groupIndex;
                }
                case NeuronGroup.Type.OUTPUT:
                {
                    outputGroups.Add(newGroup);
                    int groupIndex = outputGroups.Count - 1;

                    newGroup.index = groupIndex;

                    return groupIndex;
                }
                default:
                {
                    return NeuronGroup.INVALID_NEURON_INDEX;
                }
            }
        }

        public double GetWeightCap()
        {
            return weightCap;
        }

        //Returns double.Nan if the neuron or neuron group does not exist.
        public double GetNeuronValue(NeuronGroup.Identifier ident, uint neuronIndex)
        {
            if (!VerifyIdentifier(ident))
                return double.NaN;

            int typei = (int)ident.type;
            int indexi = (int)ident.index;

            if (neuronIndex >= allListGroups[typei][indexi].neurons.Count)
                return double.NaN;

            return allListGroups[typei][indexi].neurons[(int)neuronIndex];
        }

        //Returns double.Nan if the neuron or neuron group does not exist.
        public double GetOutputValue(uint groupIndex, uint index)
        {
            if (groupIndex >= outputGroups.Count)
                return double.NaN;

            if (index >= outputGroups[(int)groupIndex].neurons.Count)
                return double.NaN;

            return outputGroups[(int)groupIndex].neurons[(int)index];
        }

        //Returns the sum of the squared reconstruction error for the current sample.
        public double CalculateBatchError()
        {
            PropagateSignal();

            double error = 0.0;

            for (int i = 0; i < inputGroups.Count; i++)
                error += inputGroups[i].GetReconstructionError();

            for (int i = 0; i < hiddenGroups.Count; i++)
                error += hiddenGroups[i].GetReconstructionError();

            return error;
        }

        //Returns the average reconstruction error for the entire history buffer.
        public double GetBatchError()
        {
            double error = 0.0;

            foreach (List<double> sample in historyBuffer) 
            {
                SetExperience(sample);

                error += CalculateBatchError();
            }

            error /= historyBuffer.Count;

            return error;
        }

        //Returns the average reconstruction error over the past [historyBufferSize] ticks.
        public double GetOnlineError()
        {
            return averageError;
        }

        //Get neuron values of a neuron group.
        public List<double> GetNeuronValues(NeuronGroup.Identifier nGroup)
        {
            if (!VerifyIdentifier(nGroup))
                return null;

            List<double> neuronValues = new List<double>();

            for (int i = 0; i < allListGroups[(int)nGroup.type][nGroup.index].neurons.Count; i++)
                neuronValues.Add(allListGroups[(int)nGroup.type][nGroup.index].neurons[i]);

            return neuronValues;
        }

        //Get the strength of connections in a connection group.
        public List<double> GetWeights(NeuronGroup.Identifier fromGroup, NeuronGroup.Identifier toGroup)
        {
            if (!VerifyIdentifier(fromGroup) || !VerifyIdentifier(toGroup))
                return null;

            return allListGroups[(int)fromGroup.type][fromGroup.index].GetWeights(toGroup);
        }

        //Returns the Ids of all groups connected by outgoing connections to the specifed group.
        public List<NeuronGroup.Identifier> GetGroupsConnected(NeuronGroup.Identifier connectedTo)
        {
            if (!VerifyIdentifier(connectedTo))
                return null;

            return allListGroups[(int)connectedTo.type][connectedTo.index].GetGroupsConnected();
        }

        //Constructs NeuralNetwork.
        private void Construct(uint historySize, double outputNoiseMag, double weightNoiseMag, bool useNoveltyBuffer)
        {
            useNovelty = useNoveltyBuffer;

            historyBufferSize = historySize;

            outputNoiseMagnitude = outputNoiseMag;
            weightNoiseMagnitude = weightNoiseMag;
            outputNoiseRange = outputNoiseMagnitude * DOUBLE_MAGNITUDE;
            weightNoiseRange = weightNoiseMagnitude * DOUBLE_MAGNITUDE;

            weightCap = double.MaxValue;
            averageError = 0.0;

            int historyBufferSizei = (int)historyBufferSize;

            if (useNovelty)
                noveltyBuffer = new List<NoveltyBufferOccupant>(historyBufferSizei);
            else
                historyBuffer = new Queue<List<double>>(historyBufferSizei);

            errorBuffer = new Queue<double>(historyBufferSizei);

            allListGroups = new List<List<NeuronGroup>>();

            inputGroups = new List<NeuronGroup>();
            hiddenGroups = new List<NeuronGroup>();
            outputGroups = new List<NeuronGroup>();

            allListGroups.Add(inputGroups);
            allListGroups.Add(hiddenGroups);
            allListGroups.Add(outputGroups);
        }

        //Set the inputs of the neural network to a given experience.
        private void SetExperience(List<double> sample)
        {
            int dataCount = sample.Count;
            int sampleIndex = 0;

            for (int x = 0; x < inputGroups.Count; x++)
            {
                if (dataCount <= 0)
                    break;

                int neuronCount = inputGroups[x].neurons.Count;

                //If there is to little data use only what is provided.
                if (dataCount < neuronCount)
                    neuronCount = dataCount;

                for (int y = 0; y < neuronCount; y++)
                {
                    inputGroups[x].neurons[y] = sample[sampleIndex];
                    sampleIndex++;
                }

                dataCount -= neuronCount;
            }
        }

        private void UpdateOnlineError(double currentError)
        {
            //Make sure there is at least one error value.
            if (errorBuffer.Count < 1)
            {
                //Just add the new error, nothing else.
                errorBuffer.Enqueue(currentError);
                return;
            }

            //Convert the error count to a double.
            double errorCount = (double)(errorBuffer.Count);

            //If the error queue is not filled to capacity,
            //the average has to be calculated by summing up all the errors.
            if (errorBuffer.Count < historyBufferSize)
            {
                double errorSum = 0.0;

                foreach (double error in errorBuffer)
                    errorSum += error;

                averageError = errorSum / errorCount;
            }
            //errorCount does not change. Error can be calculated by removing
            //the first error and adding the new error, each over errorCount.
            else
            {
                averageError -= errorBuffer.Peek() / errorCount;
                averageError += currentError / errorCount;

                //Get rid of the old error value.
                errorBuffer.Dequeue();
            }

            //Add the new error.
            errorBuffer.Enqueue(currentError);
        }

        //Add a new experience to the novelty buffer.
        private void AddNoveltyOccupant(NoveltyBufferOccupant newOccupant, List<DistanceDescription> distDescriptions)
        {
            //Add the distance for the new occupant to each other and to itself if needed.
            for (int i = 0; i < noveltyBuffer.Count; i++)
            {
                DistanceDescription newDes = new DistanceDescription();
                newDes.distance = distDescriptions[i].distance;
                newDes.distanceOwner = newOccupant;

                TryInsertDistance(noveltyBuffer[i], newDes);
            }

            noveltyBuffer.Add(newOccupant);

            UpdateNoveltyScores();
            noveltyBuffer.Sort();
        }

        //Insert the distance if it is closer than the current farthest distance.
        private void TryInsertDistance(NoveltyBufferOccupant occupant, DistanceDescription distDesc)
        {
            //Only check if a distance cache removal is needed if the distance cache is full.
            if (occupant.distanceDescriptions.Count >= N_NEAREST)
            {
                //Don't insert the distance if it is farther than the current farthest distance.
                if (distDesc.distance > occupant.distanceDescriptions.Last.Value.distance)
                    return;

                //Check if the distance cache is full. Make room for the new distance.
                if (occupant.distanceDescriptions.Count == N_KEEP)
                    occupant.distanceDescriptions.RemoveLast();
            }

            //Insert the distance into the sorted distance list of the occupant.
            for (LinkedListNode<DistanceDescription> it = occupant.distanceDescriptions.Last; it != null; it = it.Previous)
            {
                //Check where the new distance should be placed.
                if (distDesc.distance > it.Value.distance)
                {
                    occupant.distanceDescriptions.AddAfter(it, distDesc);
                    return;
                }
            }

            //The new distance is the new nearest.
            occupant.distanceDescriptions.AddFirst(distDesc);
        }

        //Remove an experience from the novelty buffer.
        private void RemoveNoveltyOccupant(NoveltyBufferOccupant oldOccupant)
        {
            noveltyBuffer.Remove(oldOccupant);

            foreach (NoveltyBufferOccupant occupant in noveltyBuffer)
            {
                for (LinkedListNode<DistanceDescription> it = occupant.distanceDescriptions.First; it != null; it = it.Next)
                {
                    //Remove the distance if it is the distance to the removed occupant.
                    if (it.Value.distanceOwner.Equals(oldOccupant))
                    {
                        occupant.distanceDescriptions.Remove(it);
                        break;
                    }
                }

                //Recompute novelty scores for the occupant if it the novelty buffer 
                //has experiences that can fill the rest of the occupant's nearest distances.
                if (noveltyBuffer.Count > N_KEEP && occupant.distanceDescriptions.Count < N_NEAREST)
                    ComputeDistances(occupant);
            }
        }

        private void UpdateNoveltyScores()
        {
            foreach (NoveltyBufferOccupant occupant in noveltyBuffer)
            {
                int distancesToSum = occupant.distanceDescriptions.Count;

                if (distancesToSum > N_NEAREST)
                    distancesToSum = (int)N_NEAREST;

                occupant.noveltyScore = 0.0;

                LinkedListNode<DistanceDescription> distanceIterator = occupant.distanceDescriptions.First;

                for (int i = 0; i < distancesToSum; i++)
                {
                    occupant.noveltyScore += distanceIterator.Value.distance;
                    distanceIterator = distanceIterator.Next;
                }
            }
        }

        //Expensive computation of distances for an experience already in buffer.
        private void ComputeDistances(NoveltyBufferOccupant occupant)
        {
            if (noveltyBuffer.Count == 0)
                return;

            //Only need one less than the number of novelty buffer occupants.
            List<DistanceDescription> distanceDescriptions = new List<DistanceDescription>(noveltyBuffer.Count - 1);

            foreach (NoveltyBufferOccupant currentOccupant in noveltyBuffer)
            {
                //Don't include the least novel occupant.
                if (currentOccupant != occupant)
                {
                    double distance = ExpDistance(occupant.experience, currentOccupant.experience);

                    DistanceDescription newDescription = new DistanceDescription();
                    newDescription.distance = distance;
                    newDescription.distanceOwner = currentOccupant;

                    distanceDescriptions.Add(newDescription);
                }
            }

            distanceDescriptions.Sort();

            //Reset distances.
            occupant.distanceDescriptions.Clear();

            //This method is only called when noveltyBuffer.Count > N_KEEP.
            for (int i = 0; i < N_KEEP; i++)
                occupant.distanceDescriptions.AddLast(distanceDescriptions[i]);
        }

        //Makes sure a type is INPUT, HIDDEN, or OUTPUT.
        private bool VerifyType(NeuronGroup.Type type)
        {
            int typei = (int)type;

            if (typei < (int)NeuronGroup.Type.INPUT || typei > (int)NeuronGroup.Type.OUTPUT)
                return false;
            else
                return true;
        }

        //Makes sure an identifier specifies a neuron group within allListGroups.
        private bool VerifyIdentifier(NeuronGroup.Identifier ident)
        {
            if (!VerifyType(ident.type))
                return false;

            if (ident.index < 0 || ident.index >= allListGroups[(int)ident.type].Count)
                return false;

            return true;
        }

        private double ExpDistance(List<double> exp, List<double> compare)
        {
            double sum = 0.0;

            for (int i = 0; i < exp.Count; i++)
                sum += Math.Pow(exp[i] - compare[i], 2.0);

            return Math.Sqrt(sum);
        }

        //Expensive computation of novelty score for a new experiences.
        private List<DistanceDescription> ComputeNewDistances(NoveltyBufferOccupant occupant)
        {
            if (noveltyBuffer.Count == 0)
                return null;

            List<DistanceDescription> distanceDescriptions = null;

            if (noveltyBuffer.Count < historyBufferSize)
            {
                distanceDescriptions = new List<DistanceDescription>(noveltyBuffer.Count);

                foreach (NoveltyBufferOccupant currentOccupant in noveltyBuffer)
                {
                    double distance = ExpDistance(occupant.experience, currentOccupant.experience);

                    DistanceDescription newDescription = new DistanceDescription();
                    newDescription.distanceOwner = currentOccupant;
                    newDescription.distance = distance;

                    distanceDescriptions.Add(newDescription);
                }
            }
            else
            {
                //Only need one less than the number of novelty buffer occupants.
                distanceDescriptions = new List<DistanceDescription>(noveltyBuffer.Count - 1);

                //Don't include the least novel occupant in the distance cache.
                for (int i = 1; i < noveltyBuffer.Count; i++)
                {
                    NoveltyBufferOccupant currentOccupant = noveltyBuffer[i];

                    double distance = ExpDistance(occupant.experience, currentOccupant.experience);

                    DistanceDescription newDescription = new DistanceDescription();
                    newDescription.distanceOwner = currentOccupant;
                    newDescription.distance = distance;

                    distanceDescriptions.Add(newDescription);
                }
            }

            //Copy the distances for sorting. The original list order is maintained 
            //so its distances can easily be added to the other novelty buffer occupants.
            List<DistanceDescription> sortedList = new List<DistanceDescription>(distanceDescriptions.Count);

            for (int i = 0; i < distanceDescriptions.Count; i++)
                sortedList.Add(distanceDescriptions[i]);

            sortedList.Sort();

            //Find out how many elements to keep in the occupant's distance caches.
            int keepCount = sortedList.Count;
            //Find out how many elements to use for the occupant's novelty score.
            int nearestCount = sortedList.Count;

            //keepCount should be at most N_KEEP.
            if (keepCount > N_KEEP)
                keepCount = (int)N_KEEP;

            //nearestCount should be at most N_NEAREST.
            if (nearestCount > N_NEAREST)
                nearestCount = (int)N_NEAREST;

            for (int i = 0; i < keepCount; i++)
                occupant.distanceDescriptions.AddLast(sortedList[i]);

            for (int i = 0; i < nearestCount; i++)
                occupant.noveltyScore += sortedList[i].distance;

            return distanceDescriptions;
        }
    }
}
