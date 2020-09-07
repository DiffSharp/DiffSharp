(*** condition: prepare ***)
#r "../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget:RestoreSources=https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#i "nuget: https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
DiffSharp: Differentiable Functional Programming
================================================


DiffSharp is a tensor library with advanced support for [differentiable programming](https://en.wikipedia.org/wiki/Automatic_differentiation).
It is designed for use in machine learning, probabilistic programming, optimization and other domains.

DiffSharp provides advanced automatic differentiation capabilities for tensor code.
Using DiffSharp, it is possible to take advanced differentives including gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector
products. This goes far beyond the simple reverse-mode gradients of traditional tensor libraries such as TensorFlow and PyTorch,
allowing you to use the full expressive capability of the host language, including control flow, while still preserving the ability to take
advanced differentiation compositions. These can use nested
forward and reverse AD up to any level, meaning that you can compute exact higher-order derivatives or differentiate functions
that are internally making use of differentiation. Please see the [API Overview](api-overview.html) page for a list of available operations.

DiffSharp 1.0 is implemented in F# and uses PyTorch C++ tensors by default. It is tested on Linux and Windows.
The library is developed by [Atılım Güneş Baydin](https://www.cs.nuim.ie/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and other contributors. Please join us!

Current Features and Roadmap
----------------------------

The primary features of DiffSharp 1.0 are:

- A tensor programming model for F#.

- A reference backend for correctness testing.

- [PyTorch](https://pytorch.org/) backend for CUDA support and highly optimized native tensor operations.

- Nested differentiation for tensors, supporting forward and reverse AD, or any combination thereof, up to any level.

- Matrix-free Jacobian- and Hessian-vector products.

- Common optimizers and model elements including convolutions.

Quick Usage Example
-------------------

Below is a sample of using DiffSharp. You can access this sample as a [script](index.fsx) or a [.NET Interactive Jupyter Notebook](index.ipynb)
(open in [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=index.ipynb)).
*)

open DiffSharp

// A 1D tensor
let t1 = dsharp.tensor [ 0.0 .. 0.2 .. 1.0 ]

// A 2x2 tensor
let t2 = dsharp.tensor [ [ 0; 1 ]; [ 2; 2 ] ]

// Define a scalar-to-scalar function
let f (x: Tensor) = sin (sqrt x)

f (dsharp.tensor 1.2)

// Get its derivative
let df = dsharp.diff f

df (dsharp.tensor 1.2)

// Now define a vector-to-scalar function
let g (x: Tensor) = exp (x.[0] * x.[1]) + x.[2]

g (dsharp.tensor [ 0.0; 0.3; 0.1 ])

// Now compute the gradient of g
let gg = dsharp.grad g 

gg (dsharp.tensor [ 0.0; 0.3; 0.1 ])

// Compute the hessian of g
let hg = dsharp.hessian g

hg (dsharp.tensor [ 0.0; 0.3; 0.1 ])

(**
More Info and How to Cite
-------------------------

To learn more about DiffSharp, use the navigation links to the right.

If you are using DiffSharp, please raise any issues you might have [on GitHub](https://github.com/DiffSharp/DiffSharp).
We also have a [Gitter chat room](https://gitter.im/DiffSharp/DiffSharp).
If you would like to cite this library, please cite both this documentation, and use the following information:

_Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2015) Automatic differentiation and machine learning: a survey. arXiv preprint. arXiv:1502.05767_ ([link](https://arxiv.org/abs/1502.05767)) ([BibTeX](misc/adml2015.bib))

*)
