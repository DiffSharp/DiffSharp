
(**
Interoperability with Other Languages
=====================================

As F# can interoperate seamlessly with C# and other [CLI languages](http://en.wikipedia.org/wiki/List_of_CLI_languages), DiffSharp can be used with these languages as well. Your project should reference the **DiffSharp.dll** assembly, its dependencies, and also the **FSharp.Core.dll** assembly.

For C# and other languages, the **DiffSharp.Interop** namespace provides a simple way of accessing the main functionality. (Without **DiffSharp.Interop**, you can still use the regular DiffSharp namespaces, but you will need to take care of issues such as converting to and from [**FSharp.Core.FSharpFunc**](https://msdn.microsoft.com/en-us/library/ee340302.aspx) objects.)

Using DiffSharp with C#
=======================

Nested Automatic Differentiation
--------------------------------

For using the nested forward and reverse AD capability, you need to write the part of your numeric code where you need deriatives (e.g. for optimization) using the **DiffSharp.Interop.D** numeric type, the results of which you may convert later to a standard type such as **double**. In other words, for any computation you do with the **D** numeric type, you can automatically get exact derivatives.

The **DiffSharp.Interop.AD** class provides common mathematical functions (e.g. **AD.Exp**, **AD.Sin**, **AD.Pow** ) for the **D** type, similar to the use of [**System.Math**](https://msdn.microsoft.com/en-us/library/System.Math(v=vs.110).aspx) class with the **double** type and other types.

C# versions of the differentiation operations are also provided through the **DiffSharp.Interop.AD** wrapper class, which internally handles all necessary conversions to and from C# functions. The names of differentiation operations (e.g. **diff**, **grad**, **hessian** ) remain the same, but their first letters are capitalized (e.g. **AD.Diff**, **AD.Grad**, **AD.Hessian** ). Please see the [API Overview](api-overview.html) page for general information about the differentiation API.

Here is a simple example illustrating the use of the **D** type and the computation of derivatives.

    [lang=csharp]
    // Open DiffSharp interop
    using DiffSharp.Interop;

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

            // Creation and conversion of D values

            // Construct new D
            D a = new D(4.1);

            // Cast double to D
            D b = (D)4.1;

            // Cast D to double
            double c = (double)a;
        }
    }

Differentiation operations can be nested, meaning that you can compute higher-order derivatives and differentiate functions that are themselves internally making use of differentiation (also see [Nested AD](gettingstarted-nestedad.html)).

    [lang=csharp]
    using DiffSharp.Interop;

    class Program
    {
        // F(x) = Sin(x^2 - Exp(x))
        public D F(D x)
        {
            return AD.Sin(x * x - AD.Exp(x));
        }

        // G is internally using the derivative of F
        // G(x) = F'(x) / Exp(x^3)
        public D G(D x)
        {
            return AD.Diff(F, x) / AD.Exp(AD.Pow(x, 3));
        }

        // H is internally using the derivative of G
        // H(x) = Sin(G'(x) / 2)
        public D H(D x)
        {
            return AD.Sin(AD.Diff(G, x) / 2);
        }
    }

A convenient way of writing functions is to use [C# lambda expressions](https://msdn.microsoft.com/en-us/library/bb397687.aspx) with which you can define local anonymous functions.

    [lang=csharp]
    using DiffSharp.Interop;

    class Program
    {
        // F(x) = Sin(x^2 - Exp(x))
        public static D F(D x)
        {
            return AD.Sin(x * x - AD.Exp(x));
        }

        public static void Main(string[] args)
        {
            // Derivative of F(x) at x = 3
            var a = AD.Diff(F, 3);

            // This is the same with above, defining the function inline
            var b = AD.Diff(x => AD.Sin(x * x - AD.Exp(x)), 3);
        }

DiffSharp can handle nested cases such as computing the derivative of a function $f$ that takes an argument $x$, which, in turn, computes the derivative of another function $g$ nested inside $f$ that has a free reference to $x$, the argument to the surrounding function.

$$$
  \frac{d}{dx} \left. \left( x \left( \left. \frac{d}{dy} x y \; \right|_{y=3} \right) \right) \right|_{x=2}

    [lang=csharp]
    var c = AD.Diff(x => x * AD.Diff(y => x * y, 3), 2);

This allows you to write, for example, nested optimization algorithms of the form

$$$
  \mathbf{min} \left( \lambda x \; . \; (f \; x) + \mathbf{min} \left( \lambda y \; . \; g \; x \; y \right) \right)\; ,

for functions $f$ and $g$ and a gradient-based minimization procedure $\mathbf{min}$.

### Differentiation Operations

Currently the following operations are supported by **DiffSharp.Interop.AD**:

### AD.Diff
#### First derivative of a scalar-to-scalar function

Signature: `AD.Diff(Func<D,D>)`

Return value: `Func<D,D>`

For a function $f(a): \mathbb{R} \to \mathbb{R}$, this gives the derivative

$$$
  \frac{d}{da} f(a) \; .

    [lang=csharp]
    // Derivative of a scalar-to-scalar function
    var df = AD.Diff(x => AD.Sin(x * x - AD.Exp(x)));

    // Evaluate df at a point
    var v = df(3);

#### First derivative of a scalar-to-scalar function evaluated at a point

Signature:  `AD.Diff(Func<D,D>, D)`

Return value: `D`

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the derivative evaluated at $x$

$$$
  \left. \frac{d}{da} f(a) \right|_{a\; =\; x} \; .

    [lang=csharp]
    // Derivative of a scalar-to-scalar function at a point
    var v = AD.Diff(x => AD.Sin(x * x - AD.Exp(x)), 3);

### AD.Diff2

#### Second derivative of a scalar-to-scalar function

Signature: `AD.Diff2(Func<D,D>)`

Return value: `Func<D,D>`

For a function $f(a): \mathbb{R} \to \mathbb{R}$, this gives the second derivative

$$$
  \frac{d^2}{da^2} f(a) \; .

    [lang=csharp]
    // Second derivative of a scalar-to-scalar function
    var df = AD.Diff2(x => AD.Sin(x * x - AD.Exp(x)));

    // Evaluate df at a point
    var v = df(3);

#### Second derivative of a scalar-to-scalar function evaluated at a point

Signature: `AD.Diff2(Func<D,D>, D)`

Return value: `D`

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the second derivative evaluated at $x$

$$$
  \left. \frac{d^2}{da^2} f(a) \right|_{a\; =\; x} \; .

    [lang=csharp]
    // Second derivative of a scalar-to-scalar function at a point
    var v = AD.Diff2(x => AD.Sin(x * x - AD.Exp(x)), 3);

### AD.Diffn

#### N-th derivative of a scalar-to-scalar function

Signature: `AD.Diffn(Int32, Func<D,D>)`

Return value: `Func<D,D>`

For $n \in \mathbb{N}$ and a function $f(a): \mathbb{R} \to \mathbb{R}$, this gives the n-th derivative

$$$
  \frac{d^n}{da^n} f(a) \; .

    [lang=csharp]
    // Fifth derivative of a scalar-to-scalar function
    var df = AD.Diffn(5, x => AD.Sin(x * x - AD.Exp(x)));

    // Evaluate df at a point
    var v = df(3);

#### N-th derivative of a scalar-to-scalar function evaluated at a point

Signature: `AD.Diffn(Int32, Func<D,D>, D)`

Return value: `D`

For $n \in \mathbb{N}$, a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the n-th derivative evaluated at $x$

$$$
  \left. \frac{d^n}{da^n} f(a) \right|_{a\; =\; x} \; .

    [lang=csharp]
    // Fifth derivative of a scalar-to-scalar function at a point
    var v = AD.Diffn(5, x => AD.Sin(x * x - AD.Exp(x)), 3);

### AD.Grad

#### Gradient of a vector-to-scalar function

Signature: `AD.Grad(Func<D[], D>)`

Return value: `Func<D[],D[]>`

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, this gives the [gradient](http://en.wikipedia.org/wiki/Gradient)

$$$
  \nabla f = \left[ \frac{\partial f}{{\partial a}_1}, \dots, \frac{\partial f}{{\partial a}_n} \right] \; .

    [lang=csharp]
    // Gradient of a vector-to-scalar function
    var gf = AD.Grad(x => AD.Sin(x[0] * x[1]));

    // Evaluate gf at a point
    var v = gf(new D[] { 3, 2 });

#### Gradient of a vector-to-scalar function evaluated at a point

Signature: `AD.Grad(Func<D[],D>, D[])`

Return value: `D[]`

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the gradient evaluated at $\mathbf{x}$

$$$
  \left( \nabla f \right)_\mathbf{x} = \left. \left[ \frac{\partial f}{{\partial a}_1}, \dots, \frac{\partial f}{{\partial a}_n} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Gradient of a vector-to-scalar function at a point
    var v = AD.Grad(x => AD.Sin(x[0] * x[1]), new D[] { 3, 2 });

### AD.Gradv

#### Gradient-vector product (directional derivative)

Signature: `AD.Gradv(Func<D[],D>, D[], D[])`

Return value: `D`

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the [gradient-vector product](http://en.wikipedia.org/wiki/Directional_derivative) (directional derivative), that is, the dot product of the gradient of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \nabla f \right)_\mathbf{x} \cdot \mathbf{v} \; .

With AD, this value is computed efficiently in one forward evaluation of the function, without computing the full gradient.

    [lang=csharp]
    // Gradient-vector product of a vector-to-scalar function
    var v = AD.Gradv(x => AD.Sin(x[0] * x[1]), new D[] { 3, 2 }, new D[] { 5, 3 } );

### AD.Hessian

#### Hessian of a vector-to-scalar function

Signature: `AD.Hessian(Func<D[],D>)`

Return value: `Func<D[],D[,]>`

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, this gives the [Hessian matrix](http://en.wikipedia.org/wiki/Hessian_matrix)

$$$
  \mathbf{H}_f = \begin{bmatrix}
                    \frac{\partial ^2 f}{\partial a_1^2} & \frac{\partial ^2 f}{\partial a_1 \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_1 \partial a_n} \\
                    \frac{\partial ^2 f}{\partial a_2 \partial a_1} & \frac{\partial ^2 f}{\partial a_2^2} & \cdots & \frac{\partial ^2 f}{\partial a_2 \partial a_n} \\
                    \vdots  & \vdots  & \ddots & \vdots  \\
                    \frac{\partial ^2 f}{\partial a_n \partial a_1} & \frac{\partial ^2 f}{\partial a_n \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_n^2}
                    \end{bmatrix} \; .

    [lang=csharp]
    // Hessian of a vector-to-scalar function
    var hf = AD.Hessian(x => AD.Sin(x[0] * x[1]));

    // Evaluate hf at a point
    var v = hf(new D[] { 3, 2 });

#### Hessian of a vector-to-scalar function evaluated at a point

Signature: `AD.Hessian(Func<D[],D>, D[])`

Return value: `D[,]`

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the Hessian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{H}_f \right)_\mathbf{x} = \left. \begin{bmatrix}
                                           \frac{\partial ^2 f}{\partial a_1^2} & \frac{\partial ^2 f}{\partial a_1 \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_1 \partial a_n} \\
                                           \frac{\partial ^2 f}{\partial a_2 \partial a_1} & \frac{\partial ^2 f}{\partial a_2^2} & \cdots & \frac{\partial ^2 f}{\partial a_2 \partial a_n} \\
                                           \vdots  & \vdots  & \ddots & \vdots  \\
                                           \frac{\partial ^2 f}{\partial a_n \partial a_1} & \frac{\partial ^2 f}{\partial a_n \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_n^2}
                                          \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Hessian of a vector-to-scalar function at a point
    var v = AD.Hessian(x => AD.Sin(x[0] * x[1]), new D[] { 3, 2 });

### AD.Hessianv

#### Hessian-vector product

Signature: `AD.Hessianv(Func<D[],D>, D[], D[])`

Return value: `D[]`

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the [Hessian-vector product](http://en.wikipedia.org/wiki/Hessian_automatic_differentiation), that is, the multiplication of the Hessian matrix of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{H}_f \right)_\mathbf{x} \; \mathbf{v} \; .

With AD, this value is computed efficiently using one forward and one reverse evaluation of the function, in a matrix-free way (without computing the full Hessian matrix).

    [lang=csharp]
    // Hessian-vector product of a vector-to-scalar function
    var hv = AD.Hessianv(x => AD.Sin(x[0] * x[1]), new D[] { 3, 2 }, new D[] { 5, 3 });

### AD.Laplacian

#### Laplacian of a vector-to-scalar function

Signature: `AD.Laplacian(Func<D[],D>)`

Return value: `Func<D[],D>`

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the sum of second derivatives evaluated at $\mathbf{x}$

$$$
  \mathrm{tr}\left(\mathbf{H}_f \right) = \left(\frac{\partial ^2 f}{\partial a_1^2} + \dots + \frac{\partial ^2 f}{\partial a_n^2}\right) \; ,

which is the trace of the Hessian matrix.

With AD, this value is computed efficiently in a Matrix-free way, without computing the full Hessian matrix.

    [lang=csharp]
    // Laplacian of a vector-to-scalar function
    var lf = AD.Laplacian(x => AD.Sin(x[0] * x[1]));

    // Evaluate lf at a point
    var v = lf(new D[] { 3, 2 });

#### Laplacian of a vector-to-scalar function evaluated at a point

Signature: `AD.Laplacian(Func<D[],D>, D[])`

Return value: `D`

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the sum of second derivatives evaluated at $\mathbf{x}$

$$$
  \mathrm{tr}\left(\mathbf{H}_f \right)_\mathbf{x} = \left. \left(\frac{\partial ^2 f}{\partial a_1^2} + \dots + \frac{\partial ^2 f}{\partial a_n^2}\right) \right|_{\mathbf{a} \; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Laplacian of a vector-to-scalar function at a point
    var v = AD.Laplacian(x => AD.Sin(x[0] * x[1]), new D[] { 3, 2 });

### AD.Jacobian

#### Jacobian of a vector-to-vector function

Signature: `AD.Jacobian(Func<D[],D[]>)`

Return value: `Func<D[],D[,]>`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, this gives the $m$-by-$n$ [Jacobian matrix](http://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)

$$$
  \mathbf{J}_\mathbf{F} = \begin{bmatrix}
                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_1}{\partial a_n} \\
                            \vdots & \ddots & \vdots  \\
                            \frac{\partial F_m}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_n}
                            \end{bmatrix} \; .

    [lang=csharp]
    // Jacobian of a vector-to-vector function
    var jf = AD.Jacobian(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] });

    // Evaluate jf at a point
    var v = jf(new D[] { 3, 2, 4 });

#### Jacobian of a vector-to-vector function evaluated at a point

Signature: `AD.Jacobian(Func<D[],D[]>, D[])`

Return value: `D[,]`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $m$-by-$n$ Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_1}{\partial a_n} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_m}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Jacobian of a vector-to-vector function at a point
    var v = AD.Jacobian(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] }, new D[] { 3, 2, 4 });

### AD.Jacobianv

#### Jacobian-vector product

Signature: `AD.Jacobianv(Func<D[],D[]>, D[], D[])`

Return value: `D[]`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the Jacobian-vector product, that is, the matrix product of the Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} \mathbf{v} \; .
  
With AD, this value is computed efficiently in one forward evaluation of the function, in a matrix-free way (without computing the full Jacobian matrix).

    [lang=csharp]
    // Jacobian-vector product of a vector-to-vector function
    var v = AD.Jacobianv(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] }, new D[] { 3, 2, 4 }, new D[] { 1, 2, 3 });

### AD.JacobianT

#### Transposed Jacobian of a vector-to-vector function

Signature: `AD.JacobianT(Func<D[],D[]>)`

Return value: `Func<D[],D[,]>`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, this gives the $n$-by-$m$ transposed Jacobian matrix

$$$
  \mathbf{J}_\mathbf{F}^\textrm{T} = \begin{bmatrix}
                                        \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_1} \\
                                        \vdots & \ddots & \vdots  \\
                                        \frac{\partial F_1}{\partial a_n} & \cdots & \frac{\partial F_m}{\partial a_n}
                                        \end{bmatrix} \; .

    [lang=csharp]
    // Transposed Jacobian of a vector-to-vector function
    var jf = AD.JacobianT(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] });

    // Evaluate jf at a point
    var v = jf(new D[] { 3, 2, 4 });

#### Transposed Jacobian of a vector-to-vector function evaluated at a point

Signature: `AD.JacobianT(Func<D[],D[]>, D[])`

Return value: `D[,]`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $n$-by-$m$ transposed Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_1} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_1}{\partial a_n} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Transposed Jacobian of a vector-to-vector function at a point
    var v = AD.JacobianT(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] }, new D[] { 3, 2, 4 });

### AD.JacobianTv

#### Transposed Jacobian-vector product

Signature: `AD.JacobianTv(Func<D[],D[]>, D[], D[])`

Return value: `D[]`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, and $\mathbf{v} \in \mathbb{R}^m$, this gives the matrix product of the transposed Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} \mathbf{v} \; .
  
With AD, this value is computed efficiently in one forward and one reverse evaluation of the function, in a matrix-free way (without computing the full Jacobian matrix).

    [lang=csharp]
    // Transposed Jacobian-vector product of a vector-to-vector function
    var v = AD.JacobianTv(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] }, new D[] { 3, 2, 4 }, new D[] { 1, 2, 3 });

### AD.Curl

#### Curl of a vector-to-vector function

Signature: `AD.Curl(Func<D[],D[]>)`

Return value: `Func<D[],D[]>`

For a function $\mathbf{F}: \mathbb{R}^3 \to \mathbb{R}^3$ with components $F_1(a_1, a_2, a_3),\; F_2(a_1, a_2, a_3),\; F_3(a_1, a_2, a_3)$ this gives the [curl](http://en.wikipedia.org/wiki/Curl_(mathematics)), that is,

$$$
  \textrm{curl} \, \mathbf{F} = \nabla \times \mathbf{F} = \left[ \frac{\partial F_3}{\partial a_2} - \frac{\partial F_2}{\partial a_3}, \; \frac{\partial F_1}{\partial a_3} - \frac{\partial F_3}{\partial a_1}, \; \frac{\partial F_2}{\partial a_1} - \frac{\partial F_1}{\partial a_2} \right] \; .
  
    [lang=csharp]
    // Curl of a vector-to-vector function
    var cf = AD.Curl(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] });

    // Evaluate cf at a point
    var v = cf(new D[] { 3, 2, 4 });

#### Curl of a vector-to-vector function evaluated at a point

Signature: `AD.Curl(Func<D[],D[]>, D[])`

Return value: `D[]`

For a function $\mathbf{F}: \mathbb{R}^3 \to \mathbb{R}^3$ with components $F_1(a_1, a_2, a_3),\; F_2(a_1, a_2, a_3),\; F_3(a_1, a_2, a_3)$, and $\mathbf{x} \in \mathbb{R}^3$, this gives the curl evaluated at $\mathbf{x}$

$$$
  \left( \textrm{curl} \, \mathbf{F} \right)_{\mathbf{x}} = \left( \nabla \times \mathbf{F} \right)_{\mathbf{x}}= \left. \left[ \frac{\partial F_3}{\partial a_2} - \frac{\partial F_2}{\partial a_3}, \; \frac{\partial F_1}{\partial a_3} - \frac{\partial F_3}{\partial a_1}, \; \frac{\partial F_2}{\partial a_1} - \frac{\partial F_1}{\partial a_2} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Curl of a vector-to-vector function at a point
    var v = AD.Curl(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] }, new D[] { 3, 2, 4 });

### AD.Div

#### Divergence of a vector-to-vector function

Signature: `AD.Div(Func<D[],D[]>)`

Return value: `Func<D[],D>`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ with components $F_1(a_1, \dots, a_n),\; \dots, \; F_n(a_1, \dots, a_n)$, this gives the [divergence](http://en.wikipedia.org/wiki/Divergence), that is, the trace of the Jacobian matrix

$$$
  \textrm{div} \, \mathbf{F} = \nabla \cdot \mathbf{F} = \textrm{tr}\left( \mathbf{J}_{\mathbf{F}} \right) = \left( \frac{\partial F_1}{\partial a_1} + \dots + \frac{\partial F_n}{\partial a_n}\right) \; .

    [lang=csharp]
    // Divergence of a vector-to-vector function
    var df = AD.Curl(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] });

    // Evaluate df at a point
    var v = df(new D[] { 3, 2, 4 });

#### Divergence of a vector-to-vector function evaluated at a point

Signature: `AD.Div(Func<D[],D[]>, D[])`

Return value: `D`

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ with components $F_1(a_1, \dots, a_n),\; \dots, \; F_n(a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the trace of the Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \textrm{div} \, \mathbf{F} \right)_{\mathbf{x}} = \left( \nabla \cdot \mathbf{F} \right)_{\mathbf{x}} = \textrm{tr}\left( \mathbf{J}_{\mathbf{F}} \right)_{\mathbf{x}} = \left. \left( \frac{\partial F_1}{\partial a_1} + \dots + \frac{\partial F_n}{\partial a_n}\right) \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

    [lang=csharp]
    // Divergence of a vector-to-vector function at a point
    var v = AD.Curl(x => new D[] { AD.Sin(x[0] * x[1]), x[0] - x[1], x[2] }, new D[] { 3, 2, 4 });

Numerical Differentiation
-------------------------

**DiffSharp.Interop** also provides access to [numerical differentiation](gettingstarted-numericaldifferentiation.html), through the **DiffSharp.Interop.Numerical** class.

Numerical differentiation operations are used with the **double** numeric type, and the common mathematical functions can be accessed using the [**System.Math**](https://msdn.microsoft.com/en-us/library/System.Math(v=vs.110).aspx) class as usual (e.g. **Math.Exp**, **Math.Sin**, **Math.Pow** ).

Currently, the following operations are supported:

- **AD.Diff**: First derivative of a scalar-to-scalar function
- **AD.Diff2**: Second derivative of a scalar-to-scalar function
- **AD.Grad**: Gradient of a vector-to-scalar function
- **AD.Gradv**: Gradient-vector product (directional derivative)
- **AD.Hessian**: Hessian of a vector-to-scalar function
- **AD.Hessianv**: Hessian-vector product
- **AD.Laplacian**: Laplacian of a vector-to-scalar function
- **AD.Jacobian**: Jacobian of a vector-to-vector function
- **AD.JacobianT**: Transposed Jacobian of a vector-to-vector function
- **AD.Jacobianv**: Jacobian-vector product
- **AD.Curl**: Curl of a vector-to-vector function
- **AD.Div**: Divergence of a vector-to-vector function

Here are some examples:

    [lang=csharp]
    using DiffSharp.Interop;

    class Program
    {
        // A scalar-to-scalar function
        // F(x) = Sin(x^2 - Exp(x))
        public static double F(double x)
        {
            return Math.Sin(x * x - Math.Exp(x));
        }

        // A vector-to-scalar function
        // G(x1, x2) = Sin(x1 * x2)
        public static double G(double[] x)
        {
            return Math.Sin(x[0] * x[1]);
        }

        // A vector-to-vector function
        // H(x1, x2, x3) = (Sin(x1 * x2), Exp(x1 - x2), x3)
        public static double[] H(double[] x)
        {
            return new double[] { Math.Sin(x[0] * x[1]), Math.Exp(x[0] - x[1]), x[2] };
        }

        public static void Main(string[] args)
        {
            // Derivative of F(x) at x = 3
            var a = Numerical.Diff(F, 3);

            // Second derivative of F(x) at x = 3
            var b = Numerical.Diff2(F, 3);

            // Gradient of G(x) at x = (4, 3)
            var c = Numerical.Grad(G, new double[] { 4, 3 });

            // Directional derivative of G(x) at x = (4, 3) along v = (2, 5)
            var d = Numerical.Gradv(G, new double[] { 4, 3 }, new double[] { 2, 5 });

            // Hessian of G(x) at x = (4, 3)
            var e = Numerical.Hessian(G, new double[] { 4, 3 });

            // Hessian-vector product of G(x), with x = (4, 3) and v = (2, 5)
            var f = Numerical.Hessianv(G, new double[] { 4, 3 }, new double[] { 2, 5 });

            // Laplacian of G(x) at x = (4, 3)
            var g = Numerical.Laplacian(G, new double[] { 4, 3 });

            // Jacobian of H(x) at x = (5, 2, 1)
            var h = Numerical.Jacobian(H, new double[] { 5, 2, 1 });

            // Transposed Jacobian of H(x) at x = (5, 2, 1)
            var i = Numerical.JacobianT(H, new double[] { 5, 2, 1 });

            // Jacobian-vector product of H(x), with x = (5, 2, 1) and v = (2, 5, 3)
            var j = Numerical.Jacobianv(H, new double[] { 5, 2, 1 }, new double[] { 2, 5, 3 });

            // Curl of H(x) at x = (5, 2, 1)
            var k = Numerical.Curl(H, new double[] { 5, 2, 1 });

            // Divergence of H(x) at x = (5, 2, 1)
            var l = Numerical.Div(H, new double[] { 5, 2, 1 });
        }
    }
*)

