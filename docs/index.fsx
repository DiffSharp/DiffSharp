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

Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
DiffSharp: Differentiable Functional Programming
================================================

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=index.ipynb) [Script](index.fsx)

DiffSharp is a tensor library with advanced support for [differentiable programming](https://en.wikipedia.org/wiki/Automatic_differentiation).

DiffSharp is designed for use in machine learning, probabilistic programming, optimization and other domains.

Using DiffSharp, advanced differentives including gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector
products are possible. This goes beyond the simple reverse-mode gradients of traditional tensor libraries such as TensorFlow and PyTorch.
The full expressive capability of the language including control flow while still preserving the ability to take
advanced differentiation compositions. These can use nested
forward and reverse AD up to any level, meaning that you can compute exact higher-order derivatives or differentiate functions
that are internally making use of differentiation. Please see the [API Overview](api-overview.html) page for a list of available operations.

The library is developed by [Atılım Güneş Baydin](https://www.cs.nuim.ie/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and other contributors. Please join us!

> DiffSharp 1.0 is implemented in F# and the default backend uses PyTorch C++ tensors. It is tested on Linux and Windows.

Current Features and Roadmap
----------------------------

The primary features of DiffSharp 1.0 are:

- _Tensor programming model for F#_
- _Reference backend for correctness testing_
- _[PyTorch](https://pytorch.org/) backend for CUDA support and highly optimized native tensor operations_
- _Nested differentiation for tensors, supporting forward and reverse AD, or any combination thereof, up to any level_
- _Matrix-free Jacobian- and Hessian-vector products_

Please join with us to help us get the API right and ensure model development with DiffSharp is as succinct and clean as
possible. See also our [github issues](https://github.com/DiffSharp/DiffSharp/issues/). 

Quick Usage Example
-------------------
*)

open DiffSharp

// A 1D tensor
let t1 = dsharp.tensor [ 0.0 .. 0.2 .. 5.0 ]

// A 2D tensor
let t2 = dsharp.tensor [ [ 0; 1]; [2; 2 ] ]

// A scalar-to-scalar function
let f (x: Tensor) = sin (sqrt x)

f (dsharp.tensor 1.2)

// Derivative of f
let df = dsharp.diff f

df (dsharp.tensor 1.2)

// A vector-to-scalar function
let g (x: Tensor) = exp (x.[0] * x.[1]) + x.[2]

g (dsharp.tensor [ 0.0; 0.3; 0.1 ])

// Gradient of g
let gg = dsharp.grad g 

gg (dsharp.tensor [ 0.0; 0.3; 0.1 ])

// Hessian of g
let hg = dsharp.hessian g

hg (dsharp.tensor [ 0.0; 0.3; 0.1 ])

(**
More Info and How to Cite
-------------------------

If you are using DiffSharp, we would be very happy to hear about it! Please get in touch with us using email or raise any issues you might have [on GitHub](https://github.com/DiffSharp/DiffSharp). We also have a [Gitter chat room](https://gitter.im/DiffSharp/DiffSharp) that we follow.

If you would like to cite this library, please use the following information:

_Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2015) Automatic differentiation and machine learning: a survey. arXiv preprint. arXiv:1502.05767_ ([link](https://arxiv.org/abs/1502.05767)) ([BibTeX](misc/adml2015.bib))

*)
