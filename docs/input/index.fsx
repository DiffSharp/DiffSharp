(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
DiffSharp: Automatic Differentiation Library
============================================

DiffSharp is an [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation) (AD) library implemented in the F# language. It [supports C#](csharp.html) and the other [CLI languages](http://en.wikipedia.org/wiki/List_of_CLI_languages).

AD allows exact and efficient calculation of derivatives, by systematically invoking the chain rule of calculus at the elementary operator level during program execution. AD is different from [numerical differentiation](http://en.wikipedia.org/wiki/Numerical_differentiation), which is prone to truncation and round-off errors, and [symbolic differentiation](http://en.wikipedia.org/wiki/Symbolic_computation), which suffers from expression swell and cannot handle algorithmic control flow.

Using the DiffSharp library, derivative calculations (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products) can be incorporated with minimal change into existing algorithms. Operations can be nested to any level, meaning that you can compute exact higher-order derivatives and differentiate functions that are internally making use of differentiation. Please see the [API Overview](api-overview.html) page for a list of available operations.

The library is under active development by [Atılım Güneş Baydin](http://www.cs.nuim.ie/~gunes/) and [Barak A. Pearlmutter](http://bcl.hamilton.ie/~barak/) mainly for research applications in machine learning, as part of their work at the [Brain and Computation Lab](http://www.bcl.hamilton.ie/), Hamilton Institute, National University of Ireland Maynooth.

<div class="row">
    <div class="span9">
    <div class="well well-small" id="nuget" style="background-color:#F0B2B2">
        <b>As of version 0.6, DiffSharp supports nesting of AD operations.</b> This entails important changes in the library structure. Please see the <a href="https://github.com/DiffSharp/DiffSharp/releases">release notes</a> to learn about the changes and how you can update your code.
    </div>
    </div>
</div>

How to Get
----------

You can install the library via NuGet. You can also download the source code or the binaries of the latest release <a href="https://github.com/DiffSharp/DiffSharp/releases">on GitHub</a>.

<div class="row">
    <div class="span1"></div>
    <div class="span7">
    <div class="well well-small" id="nuget">
        The DiffSharp library <a href="https://www.nuget.org/packages/diffsharp">is available on NuGet</a>. To install, run the following command in the <a href="http://docs.nuget.org/docs/start-here/using-the-package-manager-console">Package Manager Console</a>:
        <pre>PM> Install-Package DiffSharp</pre>
    </div>
    </div>
    <div class="span1"></div>
</div>

Quick Usage Example
-------------------
*)

// Use mixed mode nested AD
open DiffSharp.AD

// A scalar-to-scalar function
let f x = sin (sqrt x)

// Derivative of f
let df = diff f

// A vector-to-scalar function
let g (x:_[]) = exp (x.[0] * x.[1]) + x.[2]

// Gradient of g
let gg = grad g 

// Hessian of g
let hg = hessian g

(**
More Info and How to Cite
-------------------------

If you are using DiffSharp and would like to cite it, please use the following information:

_Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2015) Automatic differentiation and machine learning: a survey. arXiv preprint. arXiv:1502.05767_ ([link](http://arxiv.org/abs/1502.05767)) ([BibTeX](misc/adml2015.bib))

For a quick overview of AD and other differentiation methods, you can refer to our [recent poster](http://www.cs.nuim.ie/~gunes/files/AGBaydinICML2014Poster.pdf) for the AutoML workshop at the International Conference on Machine Learning 2014. For in-depth material, you can check our [publications page](http://www.bcl.hamilton.ie/publications/) and the [autodiff.org](http://www.autodiff.org/) website.

If you are using DiffSharp, we would be very happy to put a link to your work on this page.

Future Releases
---------------

We are working on the following features:

- Improved Hessian calculations exploiting structure (e.g. sparsity)
- AD via source code transformation, using code quotations
- Compiling to GPU

*)
