(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Reference.dll"
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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=api-overview.ipynb) [Script](api-overview.fsx)

API Overview
============

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

* [The `dsharp.*` API](/reference/diffsharp-dsharp.html)

* [Tensors](/reference/diffsharp-tensor.html)

* [Models](/reference/diffsharp-model.html)

* [Optimizers](/reference/diffsharp-optim.html)

* [Distributions](/reference/diffsharp-distributions.html)

* [Data and Data Loaders](/reference/diffsharp-data.html)

*)

