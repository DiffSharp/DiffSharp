(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.1"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Reference.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"


Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
DiffSharp: Differentiable Functional Programming
================================================

DiffSharp is a tensor library with advanced support for [differentiable programming](https://en.wikipedia.org/wiki/Automatic_differentiation).
It is designed for use in machine learning, probabilistic programming, optimization and other domains.

DiffSharp provides advanced automatic differentiation capabilities for tensor code, making it possible to use derivative-taking operations, including gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector
products, as higher-order function compositions. This goes far beyond the standard reverse-mode gradients of traditional tensor libraries such as TensorFlow and PyTorch, allowing the use of nested
forward and reverse differentiation up to any level, meaning that you can compute higher-order derivatives efficiently or differentiate functions
that are internally making use of differentiation. Please see [API Overview](api-overview.html) for a list of available operations.

DiffSharp 1.0 is implemented in F# and uses PyTorch C++ tensors (without the derivative computation graph) as the default raw-tensor backend. It is tested on Linux and Windows.
DiffSharp is developed by [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and other contributors, having started as a project supervised by [Barak Pearlmutter](https://scholar.google.com/citations?user=AxFrw0sAAAAJ&hl=en) and [Jeffrey Siskind](https://scholar.google.com/citations?user=CgSBtPYAAAAJ&hl=en). Please join us!

**The library and documentation are undergoing development.**

Current features and roadmap
----------------------------

The primary features of DiffSharp 1.0 are:

- A tensor programming model for F#.

- A reference backend for correctness testing.

- [PyTorch](https://pytorch.org/) backend for CUDA support and highly optimized native tensor operations.

- Nested differentiation for tensors, supporting forward and reverse AD, or any combination thereof, up to any level.

- Matrix-free Jacobian- and Hessian-vector products.

- Common optimizers and model elements including convolutions.

- Probability distributions.

Quick usage example
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
More information
-------------------------

To learn more about DiffSharp, use the navigation links to the left.

If you are using DiffSharp, please raise any issues you might have [on GitHub](https://github.com/DiffSharp/DiffSharp).
We also have a [Gitter chat room](https://gitter.im/DiffSharp/DiffSharp).
If you would like to cite this library, please use the following information:

_Baydin, A.G., Pearlmutter, B.A., Radul, A.A. and Siskind, J.M., 2017. Automatic differentiation in machine learning: a survey. The Journal of Machine Learning Research, 18(1), pp.5595-5637._ ([link](https://arxiv.org/abs/1502.05767))

*)
