(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
DiffSharp: Automatic Differentiation Library
============================================

DiffSharp is an [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation) (AD) library implemented in the F# language.

AD allows exact and efficient calculation of derivatives, by systematically applying the chain rule of calculus at the elementary operator level. AD is different from [numerical differentiation](http://en.wikipedia.org/wiki/Numerical_differentiation), which is prone to truncation and round-off errors, and [symbolic differentiation](http://en.wikipedia.org/wiki/Symbolic_computation), which is exact but not efficient for run-time calculations and can only handle closed-form mathematical expressions.

Using the DiffSharp library, derivative calculations (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products) can be incorporated with minimal change into existing algorithms. Please see the [API Overview](api-overview.html) page for a list of available operations.

The library is under active development by [Atılım Güneş Baydin](http://www.cs.nuim.ie/~gunes/) and [Barak A. Pearlmutter](http://bcl.hamilton.ie/~barak/) mainly for research applications in machine learning, as part of their work at the [Brain and Computation Lab](http://www.bcl.hamilton.ie/), Hamilton Institute, National University of Ireland Maynooth.

How to Get
----------

You can install the library via NuGet. You can also download the source code or the binaries of the latest release <a href="https://github.com/gbaydin/DiffSharp/releases">on GitHub</a>.

<div class="row">
    <div class="span1"></div>
    <div class="span6">
    <div class="well well-small" id="nuget">
        The DiffSharp library <a href="https://www.nuget.org/packages/diffsharp">is available on NuGet</a>. To install, run the following command in the <a href="http://docs.nuget.org/docs/start-here/using-the-package-manager-console">Package Manager Console</a>:
        <pre>PM> Install-Package DiffSharp</pre>
    </div>
    </div>
    <div class="span1"></div>
</div>

Future Releases
---------------

We are working on the following features for the next release:

- Handling of nested AD operations
- Improved Hessian calculations exploiting structure (e.g. sparsity)
- AD via source code transformation, using code quotations
- Integration with [Math.NET Numerics](http://numerics.mathdotnet.com/) vectors and matrices

Quick Usage Example
-------------------
*)

// Use forward mode AD
open DiffSharp.AD.Forward

// A scalar-to-scalar function
let f x = 
    sin (sqrt x)

// Derivative of f
let df = diff f

// Value of the derivative of f at a point
let df2 = df 2.

(**
More Info and How to Cite
-------------------------

For a quick overview of AD and other differentiation methods, you can refer to our [recent poster](http://www.cs.nuim.ie/~gunes/files/AGBaydinICML2014Poster.pdf) or [article](http://arxiv.org/abs/1404.7456) for the AutoML workshop at the International Conference on Machine Learning 2014. For in-depth material, you can check our [publications page](http://www.bcl.hamilton.ie/publications/) and the [autodiff.org](http://www.autodiff.org/) website.

We are writing an article about this library and its usage and we hope to get it ready soon. In the meantime, if you are using DiffSharp and would like to cite it, please use the following information:

_Baydin, A. G. and Pearlmutter, B. A. (2014). Diffsharp: Automatic Differentiation Library. http://gbaydin.github.io/DiffSharp/index.html._

> @misc{baydin2014diff,<br>
>   title={DiffSharp: Automatic Differentiation Library},<br>
>   author={Baydin, A. G. and Pearlmutter, B. A.},<br>
>   howpublished={\url{http://gbaydin.github.io/DiffSharp/index.html}},<br>
>   year={2014},<br>
> }<br>


If you are using DiffSharp, we would be very happy to link to your research or work on this page.

*)
