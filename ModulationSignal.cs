using System.Collections.Generic;

namespace Raahn
{
    public class ModulationSignal
    {
        //-1 to obtain passive modulation from ModulationSignal.GetSignal
        public const int INVALID_INDEX = -1;
        //Has no effect when multiplying.
        private const double BENIGN_MODULATION = 1.0;

        private static List<double> modulations = new List<double>();

        //Returns the index of the signal.
        public static uint AddSignal()
        {
            modulations.Add(BENIGN_MODULATION);
            return (uint)(modulations.Count - 1);
        }

        //Returns the index of the signal.
        public static uint AddSignal(double defaultValue)
        {
            modulations.Add(defaultValue);
            return (uint)(modulations.Count - 1);
        }

		//If the modulation does not exist, the default modulation is returnned.
        public static double GetSignal(int index)
        {
			if (index < 0 || index >= modulations.Count)
                return BENIGN_MODULATION;
			else
            	return modulations[index];
        }

        public static void SetSignal(uint index, double value)
        {
            if (index >= modulations.Count)
                return;

            modulations[(int)index] = value;
        }
    }
}