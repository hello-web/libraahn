using System;

namespace Raahn
{
    public class Activation
    {
        public static double Logistic(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

		//Takes the already computed value of sigmoid.
        public static double LogisticDerivative(double x)
        {
            return x * (1.0 - x);
        }
    }
}