(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
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
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=documentation-extensions.ipynb)&emsp;
[![Script](img/badge-script.svg)](documentation-extensions.fsx)&emsp;
[![Script](img/badge-notebook.svg)](documentation-extensions.ipynb)

Extending DiffSharp
===================

DiffSharp provides most of the essential operations found in tensor libraries such as [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/). All differentiable operations support the forward, reverse, and nested differentiation modes. 

When implementing new operations, you should prefer to implement these as compositions of existing DiffSharp `Tensor` operations, which would give you differentiability out of the box.

In the unlikely case where you need to extend DiffSharp with a completely new differentiable operation that cannot be implemented as a composition of existing operations, you can use the provided extension API.

Simple elementwise ops
----------------------

General ops
-----------

*)

