using System;

namespace Raahn
{
    class Tester
    {
        private delegate void TestFunctionType();

        private const string NO_ACTION = "Invalid action or no action chosen.";

        private readonly string[] options = 
        {
            "xor"
        };

        private readonly TestFunctionType[] tests = 
        {
            XorTest
        };

        private const int EXIT_S = 0;
        private const int EXIT_F = 1;

        public int Execute(string[] args)
        {
            if (args.Length < 1)
                return EXIT_F;

            int actionIndex = -1;

            for (int i = 0; i < options.Length; i++)
            {
                if (options[i].Equals(args[0]))
                    actionIndex = i;
            }

            if (actionIndex < 0)
            {
                Console.WriteLine(NO_ACTION);
                return EXIT_F;
            }

            tests[actionIndex]();

            return EXIT_S;
        }

        private static void XorTest()
        {
            uint inputCount = 2;
            uint hiddenCount = 3;
            uint outputCount = 1;
            int epochs = 10000;
            bool useBias = true;

            double learningRate = 1.0;

            double[][] inputs = new double[4][]
            {
                new double[] { 0.0, 0.0 },
                new double[] { 1.0, 0.0 },
                new double[] { 0.0, 1.0 },
                new double[] { 1.0, 1.0 }
            };

            double[] labels = 
            {
                0.0,
                1.0,
                1.0,
                0.0
            };

            uint iToHSig = ModulationSignal.AddSignal();
            uint hToOSig = ModulationSignal.AddSignal();

            NeuralNetwork ann = new NeuralNetwork(learningRate);

            NeuronGroup.Identifier inputGroup = new NeuronGroup.Identifier();
            inputGroup.type = NeuronGroup.Type.INPUT;

            NeuronGroup.Identifier hiddenGroup = new NeuronGroup.Identifier();
            hiddenGroup.type = NeuronGroup.Type.HIDDEN;

            NeuronGroup.Identifier outputGroup = new NeuronGroup.Identifier();
            outputGroup.type = NeuronGroup.Type.OUTPUT;

            inputGroup.index = ann.AddNeuronGroup(inputCount, inputGroup.type);
            hiddenGroup.index = ann.AddNeuronGroup(hiddenCount, hiddenGroup.type);
            outputGroup.index = ann.AddNeuronGroup(outputCount, outputGroup.type);

            ann.ConnectGroups(inputGroup, hiddenGroup, TrainingMethod.HebbianTrain, iToHSig, useBias);
            ann.ConnectGroups(hiddenGroup, outputGroup, TrainingMethod.HebbianTrain, hToOSig, useBias);

            for (int i = 0; i < epochs; i++)
            {
                for (int x = 0; x < inputs.Length; x++)
                {
                    ann.SetInputs((uint)inputGroup.index, inputs[x]);

                    ann.PropagateSignal();

                    double guess = ann.GetNeuronValue(outputGroup, 0);
                    double modulation = ((1.0 - Math.Abs(labels[x] - guess)) * 2.0) - 1.0;

                    ModulationSignal.SetSignal(iToHSig, modulation);
                    ModulationSignal.SetSignal(hToOSig, modulation);

                    ann.Train();
                }
            }

            for (int x = 0; x < inputs.Length; x++)
            {
                ann.SetInputs((uint)inputGroup.index, inputs[x]);
                ann.PropagateSignal();

                Console.WriteLine(ann.GetNeuronValue(outputGroup, 0));
                Console.WriteLine();
            }

            Console.WriteLine("Done.");
        }
    }
}
