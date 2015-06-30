using System;

namespace Raahn
{
    public class Activation
    {
        public static double Logistic(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double LogisticDerivative(double x)
        {
            double l = Logistic(x);
            return l * (1.0 - l);
        }
    }
}