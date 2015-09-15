using System;
using DiffSharp.Interop.Float64;

namespace InteropTest
{
    class Program
    {

        // F(x) = Sin(x^2 - Exp(x))
        static D F(D x)
        {
            return AD.Sin(x * x - AD.Exp(x));
        }

        // G is internally using the derivative of F
        // G(x) = F'(x) / Exp(x^3)
        static D G(D x)
        {
            return AD.Diff(F, x) / AD.Exp(AD.Pow(x, 3));
        }

        // H is internally using the derivative of G
        // H(x) = Sin(G'(x) / 2)
        static D H(D x)
        {
            return AD.Sin(AD.Diff(G, x) / 2);
        }

        static void Main(string[] args)
        {

            // Derivative of F(x) at x = 3
            var a = AD.Diff(F, 3);

            // This is the same with above, defining the function inline
            var b = AD.Diff(x => AD.Sin(x * x - AD.Exp(x)), 3);


            Console.WriteLine(H(b));
            Console.ReadKey();
        }
    }
}
