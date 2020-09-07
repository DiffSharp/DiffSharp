(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Torch.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget:RestoreSources=https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#i "nuget: https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"

Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=getting-started-torch.ipynb) [Script](getting-started-torch.fsx)

Getting Started with DiffSharp and Torch
=========

To use the Torch backend for DiffSharp, reference one of

* [`DiffSharp-cpu`](https://www.nuget.org/packages/DiffSharp-cpu) - This includes the Torch backend using CPU only.

* [`DiffSharp-cuda-linux`](https://www.nuget.org/packages/DiffSharp-cuda-linux), [`DiffSharp-cuda-windows`](https://www.nuget.org/packages/DiffSharp-cuda-windows) - These include the Torch CPU/GPU backend for Linux and Windows respectively. Large download.

Then use:
*)

open DiffSharp

dsharp.config(backend=Backend.Torch)

let t = dsharp.tensor [ 0 .. 10 ]

(**
Now examine the device and backend:
*)

let device = t.device
let backend = t.backend

(*** include-fsi-output ***)
(** 

To move a tensor to the GPU use the following:

    let t2 = t.move(Device.GPU)
*)
