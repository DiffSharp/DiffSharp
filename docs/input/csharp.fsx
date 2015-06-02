
(**
Interoperability with Other Languages
=====================================

As F# can interoperate seamlessly with C# and other [CLI languages](http://en.wikipedia.org/wiki/List_of_CLI_languages), DiffSharp can be used with these languages as well. Your project should reference the **DiffSharp.dll** assembly, its dependencies, and also the **FSharp.Core.dll** assembly.

For C# and other languages, the **DiffSharp.Interop** namespace provides a simple way of accessing the main functionality. (Without **DiffSharp.Interop**, you can still use the regular DiffSharp namespaces, but you will need to take care of issues such as converting to and from [**FSharp.Core.FSharpFunc**] objects (https://msdn.microsoft.com/en-us/library/ee340302.aspx).)

Using DiffSharp with C#
=======================

Nested Automatic Differentiation
--------------------------------

For using the nested forward and reverse AD capability, you need to write the part of your numeric code where you need deriatives (e.g. for optimization) using the **DiffSharp.Interop.D** numeric type, the results of which you may convert later to an integral type such as **double**. In other words, for any computation you do with the **D** numeric type, you can automatically get exact derivatives.

The **DiffSharp.Interop.AD** class provides common mathematical functions (e.g. **AD.Exp**, **AD.Sin**, **AD.Pow** ) for the **D** type, similar to the use of [**System.Math**](https://msdn.microsoft.com/en-us/library/System.Math(v=vs.110).aspx) class with the **double** type and other types.

C# versions of the differentiation operations are also provided through the **DiffSharp.Interop.AD** wrapper class, which internally handles conversions to and from C# functions. The names of differentiation operations (e.g. **diff**, **grad**, **hessian** ) remain the same, only their first letters are capitalized (e.g. **AD.Diff**, **AD.Grad**, **AD.Hessian** ). Please see the [API Overview](api-overview.html) page for more information.


    [lang=csharp]
    // Open DiffSharp interop
    using DiffSharp.Interop;

    class Program
    {
        // Define a function whose derivative you need
        // F(x) = sin(x^2 - exp(x))
        public static D F(D x)
        {
            return AD.Sin(x * x - AD.Exp(x));
        }

        public static void Main(string[] args)
        {
            // You can compute the value of the derivative of F at a point
            D da = AD.Diff(F, 2.3);

            // Or, you can generate a derivative function
            // dF is the derivative function of F
            var dF = AD.Diff(F);

            // Evaluate the derivative function at different points
            D db = dF(2.3);
            D dc = dF(1.4);

            // Creation and conversion of D values

            // Construct new D
            D a = new D(4.1);

            // Cast double to D
            D b = (D)4.1;

            // Cast D to double
            double c = (double)a;

        }
    }

Differentiation operations can be nested, meaning that you can compute higher-order derivatives and differentiate functions that are internally making use of differentiation (also see [Nested AD](gettingstarted-nestedad.html)).



Currently the following operations are supported by **DiffSharp.Interop.AD**:

- **AD.Diff**: First derivative of a scalar-to-scalar function
- **AD.Diff2**: Second derivative of a scalar-to-scalar function
- **AD.Diffn**: N-th derivative of a scalar-to-scalar function
- **AD.Grad**: Gradient of a vector-to-scalar function
- **AD.Gradv**: Gradient-vector product (directional derivative)
- **AD.Laplacian**: Laplacian of a vector-to-scalar function
- **AD.Jacobian**: Jacobian of a vector-to-vector function
- **AD.JacobianT**: Transposed Jacobian of a vector-to-vector function
- **AD.Jacobianv**: Jacobian-vector product
- **AD.JacobianTv**: Transposed Jacobian-vector product
- **AD.Hessian**: Hessian of a vector-to-scalar function
- **AD.Hessianv**: Hessian-vector product
- **AD.Curl**: Curl of a vector-to-vector function
- **AD.Div**: Divergence of a vector-to-vector function

Numerical Differentiation
-------------------------

**DiffSharp.Interop** also provides 
*)

