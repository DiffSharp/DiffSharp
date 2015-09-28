using DiffSharp.Interop.Float64;

class Program
{
        
    // Define a function whose derivative you need
    // F(x) = Sin(x^2 - Exp(x))
    public static D F(D x)
    {
        return AD.Sin(x * x - AD.Exp(x));
    }

    public static void Main(string[] args)
    {
        // You can compute the value of the derivative of F at a point
        D da = AD.Diff(F, 2.3);

        // Or, you can generate a derivative function which you may use for many evaluations
        // dF is the derivative function of F
        var dF = AD.Diff(F);

        // Evaluate the derivative function at different points
        D db = dF(2.3);
        D dc = dF(1.4);

        // Construction and casting of D (scalar) values
        // Construct new D
        D a = new D(4.1);
        // Cast double to D
        D b = (D)4.1;
        // Cast D to double
        double c = (double)b;

        // Construction and casting of DV (vector) values
        // Construct new DV
        DV va = new DV(new double[] { 1, 2, 3 });
        // Cast double[] to DV
        double[] vaa = new double[] { 1, 2, 3 };
        DV vb = (DV)vaa;
        // Cast DV to double[]
        double[] vc = (double[])vb;

        // Construction and casting of DM (matrix) values
        // Construct new DM
        DM ma = new DM(new double[,] { { 1, 2 }, { 3, 4 } });
        // Cast double[,] to DM
        double[,] maa = new double[,] { { 1, 2 }, { 3, 4 } };
        DM mb = (DM)maa;
        // Cast DM to double[,]
        double[,] mc = (double[,])mb;
    }
}
