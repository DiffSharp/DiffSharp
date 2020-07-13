(*** condition: prepare ***)
#r "../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget:RestoreSources=https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#i "nuget: https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{package-version}}"
#endif // IPYNB

(**
DiffSharp: Differentiable Functional Programming
================================================

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dsyme/DiffSharp/gh-pages?filepath=notebooks/index.ipynb)

DiffSharp is a functional [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) tensor-based library.

AD allows exact and efficient calculation of derivatives, by systematically invoking the chain rule of calculus at the elementary operator
level during program execution. AD is different from [numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation),
which is prone to truncation and round-off errors, and [symbolic differentiation](https://en.wikipedia.org/wiki/Symbolic_computation), which
is affected by expression swell and cannot fully handle algorithmic control flow.

Using the DiffSharp library, differentiation (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector
products) is applied using higher-order functions, that is, functions which take other functions as arguments. Your functions can use
the full expressive capability of the language including control flow. DiffSharp allows composition of differentiation using nested
forward and reverse AD up to any level, meaning that you can compute exact higher-order derivatives or differentiate functions
that are internally making use of differentiation. Please see the [API Overview](api-overview.html) page for a list of available operations.

The library is developed by [Atılım Güneş Baydin](https://www.cs.nuim.ie/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and contributors for applications in machine learning.

DiffSharp is implemented in F#. It is tested on Linux and Windows.

> Version 1.0 is a reimplementation as a tensor library using LibTorch as the primary backend.

Current Features and Roadmap
----------------------------

The primary features of DiffSharp 1.0 are:

- _Functional nested differentiation for tensor primitives, supporting forward and reverse AD, or any combination thereof, up to any level_
- _Matrix-free Jacobian- and Hessian-vector products_
- _[PyTorch](https://pytorch.org/) backend for highly optimized native tensor operations_

See also our [github issues](https://github.com/DiffSharp/DiffSharp/issues/)

Please join with us to help us get the API right and ensure model development with DiffSharp is as succinct and clean as possible/

Quick Usage Example
-------------------
*)

// Use mixed mode nested AD
open DiffSharp

// A scalar-to-scalar function
let f (x: Tensor) = sin (sqrt x)

// Derivative of f
let df = dsharp.diff f

// A vector-to-scalar function
let g (x: Tensor) = exp (x.[0] * x.[1]) + x.[2]

// Gradient of g
let gg = dsharp.grad g 

// Hessian of g
let hg = dsharp.hessian g

(**
More Info and How to Cite
-------------------------

If you are using DiffSharp, we would be very happy to hear about it! Please get in touch with us using email or raise any issues you might have [on GitHub](https://github.com/DiffSharp/DiffSharp). We also have a [Gitter chat room](https://gitter.im/DiffSharp/DiffSharp) that we follow.

If you would like to cite this library, please use the following information:

_Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2015) Automatic differentiation and machine learning: a survey. arXiv preprint. arXiv:1502.05767_ ([link](https://arxiv.org/abs/1502.05767)) ([BibTeX](misc/adml2015.bib))

*)
