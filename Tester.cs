using System;

namespace Raahn
{
    class Tester
    {
        private delegate void TestFunctionType();

        private const string NO_ACTION = "Invalid action or no action chosen.";

        private readonly string[] options = 
        {
            "or", "xor"
        };

        private readonly TestFunctionType[] tests = 
        {
            OrTest, XorTest
        };

        private const int EXIT_S = 0;
        private const int EXIT_F = 1;

        //Not an entry point with default build.
        //Switch to executable to use as entry point.
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

            Console.WriteLine("Done.");

            return EXIT_S;
        }

        private static void OrTest()
        {
            uint inputCount = 2;
            uint outputCount = 1;
            int epochs = 1000;
            int delay = 0;
            bool useBias = true;

            double learningRate = 0.1;

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
                1.0
            };

            uint iToOSig = ModulationSignal.AddSignal();

            NeuralNetwork ann = new NeuralNetwork(learningRate);

            NeuronGroup.Identifier inputGroup = new NeuronGroup.Identifier();
            inputGroup.type = NeuronGroup.Type.INPUT;

            NeuronGroup.Identifier outputGroup = new NeuronGroup.Identifier();
            outputGroup.type = NeuronGroup.Type.OUTPUT;

            inputGroup.index = ann.AddNeuronGroup(inputCount, inputGroup.type);
            outputGroup.index = ann.AddNeuronGroup(outputCount, outputGroup.type);

            ann.ConnectGroups(inputGroup, outputGroup, TrainingMethod.HebbianTrain, (int)iToOSig, learningRate, useBias);

            for (int i = 0; i < epochs; i++)
            {
                for (int x = 0; x < inputs.Length; x++)
                {
                    ann.SetInputs((uint)inputGroup.index, inputs[x]);

                    ann.PropagateSignal();

                    double guess = ann.GetNeuronValue(outputGroup, 0);
                    double modulation = ((1.0 - Math.Abs(labels[x] - guess)) * 2.0) - 1.0;

                    ModulationSignal.SetSignal(iToOSig, modulation);

                    //Print verbose.
                    Console.WriteLine(String.Format("Training example {0}, epoch {1}", x, i));
                    Console.WriteLine("Output:{0:0.0000}, Modulation:{1:0.0000}\n", guess, modulation);

                    ann.DisplayWeights();

                    Console.WriteLine();

                    if (delay > 0)
                        System.Threading.Thread.Sleep(delay);

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
        }

        private static void XorTest()
        {
            uint inputCount = 2;
            uint hiddenCount = 3;
            uint outputCount = 1;
            int epochs = 10000;
            bool useBias = true;

            double learningRate = 0.1;

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

            ann.ConnectGroups(inputGroup, hiddenGroup, TrainingMethod.HebbianTrain, (int)iToHSig, learningRate, useBias);
            ann.ConnectGroups(hiddenGroup, outputGroup, TrainingMethod.HebbianTrain, (int)hToOSig, learningRate, useBias);

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
        }
    }
}
