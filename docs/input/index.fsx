(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
DiffSharp: Automatic Differentiation Library
============================================

DiffSharp is an [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation) (AD) library. 

AD allows exact and efficient calculation of derivatives, by systematically invoking the chain rule of calculus at the elementary operator level during program execution. AD is different from [numerical differentiation](http://en.wikipedia.org/wiki/Numerical_differentiation), which is prone to truncation and round-off errors, and [symbolic differentiation](http://en.wikipedia.org/wiki/Symbolic_computation), which is affected by expression swell and cannot fully handle algorithmic control flow.

Using the DiffSharp library, derivative calculations (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products) can be incorporated with minimal change into existing algorithms. Diffsharp supports nested forward and reverse AD up to any level, meaning that you can compute exact higher-order derivatives or differentiate functions that are internally making use of differentiation. Please see the [API Overview](api-overview.html) page for a list of available operations.

The library is under active development by [Atılım Güneş Baydin](http://www.cs.nuim.ie/~gunes/) and [Barak A. Pearlmutter](http://bcl.hamilton.ie/~barak/) mainly for research applications in machine learning, as part of their work at the [Brain and Computation Lab](http://www.bcl.hamilton.ie/), Hamilton Institute, National University of Ireland Maynooth.

DiffSharp is implemented in the F# language and [can be used from C#](csharp.html) and the [other languages](http://en.wikipedia.org/wiki/List_of_CLI_languages) running on Mono or the .Net Framework, targeting the 64 bit platform. It is tested on Linux and Windows. We are working on interfaces/ports to other languages.

<div class="row">
    <div class="span9">
    <div class="well well-small" id="nuget" style="background-color:#C6AEC7">
        Version 0.7 is a reimplementation of the library with support for <b>linear algebra primitives, BLAS/LAPACK, 32- and 64-bit precision, and different CPU/GPU backends.</b> Please see the <a href="https://github.com/DiffSharp/DiffSharp/releases">release notes</a> to learn about the changes and how you can move your code to this version. (If you still need pre-0.7 documentation, <a href="http://diffsharp.github.io/DiffSharp/0.6.3/">here they are</a>).
    </div>
    </div>
</div>


Current Features and Roadmap
----------------------------

The following features are up and running:

- _Nested AD with linear algebra primitives, supporting forward and reverse AD, or any combination thereof, up to any level_
- _Matrix-free Jacobian- and Hessian-vector products_
- _[OpenBLAS](https://github.com/xianyi/OpenBLAS/wiki) backend for highly optimized native BLAS and LAPACK operations_
- _Parallel implementations of non-BLAS operations (e.g. Hadamard products, matrix transpose)_
- _Support for 32- and 64-bit floating point precision (32 bit float operations run significantly faster on many systems)_

We are working on the following features for the next release:

- _GPU backend using CUDA/OpenCL_
- _Improved Hessian calculations exploiting sparsity structure (e.g. matrix-coloring)_
- _AD via syntax tree transformation, using code quotations_

At this point we are debugging algorithmic complexity and the APIs. We are hoping the community will help us get the API right and ensure that the latest models can make use of DiffSharp as succinctly and as cleanly as possible, which would make it convenient to use in production.

How to Get
----------

Please see the [download page](download.html) for installation instructions for Linux and Windows.

Quick Usage Example
-------------------
*)

// Use mixed mode nested AD
open DiffSharp.AD.Float32

// A scalar-to-scalar function
let f x = sin (sqrt x)

// Derivative of f
let df = diff f

// A vector-to-scalar function
let g (x:DV) = exp (x.[0] * x.[1]) + x.[2]

// Gradient of g
let gg = grad g 

// Hessian of g
let hg = hessian g

(**
More Info and How to Cite
-------------------------

If you are using DiffSharp, we would be very happy to hear about it! Please get in touch with us using email or raise any issues you might have [on GitHub](http://github.com/DiffSharp/DiffSharp). We also have a [Gitter chat room](https://gitter.im/DiffSharp/DiffSharp) that we follow.

If you would like to cite this library, please use the following information:

_Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2015) Automatic differentiation and machine learning: a survey. arXiv preprint. arXiv:1502.05767_ ([link](http://arxiv.org/abs/1502.05767)) ([BibTeX](misc/adml2015.bib))

You can also check our [**recent poster**](http://www.cs.nuim.ie/~gunes/files/ICML2015-MLOSS-Poster-A0.pdf) for the [Machine Learning Open Source Software Workshop](http://mloss.org/workshop/icml15/) at the International Conference on Machine Learning 2015. For in-depth material, you can check our [publications page](http://www.bcl.hamilton.ie/publications/) and the [autodiff.org](http://www.autodiff.org/) website. 

Other sources:

- [Introduction to Automatic Differentiation](http://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/) by Alexey Radul
- [Automatic Differentiation: The most criminally underused tool in the potential machine learning toolbox?](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/) by Justin Domke

*)
