﻿(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.Torch.dll"
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
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# API Overview

The DiffSharp API is designed to be similar to [the PyTorch Python API](https://pytorch.org/docs/stable/index.html) through very similar
naming and idioms, and where elements have similar names the PyTorch documentation can generally be used as a guide.

* The `cref:T:DiffSharp.dsharp` API

* The `cref:T:DiffSharp.Tensor` type

* [Models](/reference/diffsharp-model.html)

* [Optimizers](/reference/diffsharp-optim.html)

* [Distributions](/reference/diffsharp-distributions.html)

* [Data and Data Loaders](/reference/diffsharp-data.html)

* [Pyplot Plotting utilities](/reference/diffsharp-util-pyplot.html)

*)
