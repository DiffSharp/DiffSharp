(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // IPYNB

(*** condition: fsx ***)
#if FSX
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
// Make sure to update the parts of the native load path related to version number ([cpu](https://www.nuget.org/packages/libtorch-cpu/),[gpu](https://www.nuget.org/packages/libtorch-cuda-11.1-win-x64/) and cpu/gpu depending on your backend choice.
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.8.0.7/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.8.0.7/runtimes/win-x64/native/torch_cpu.dll")
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX

(*** condition: ipynb ***)
#if IPYNB
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
// Make sure to update the parts of the native load path related to version number ([cpu](https://www.nuget.org/packages/libtorch-cpu/),[gpu](https://www.nuget.org/packages/libtorch-cuda-11.1-win-x64/) and cpu/gpu depending on your backend choice.
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.8.0.7/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.8.0.7/runtimes/win-x64/native/torch_cpu.dll")

// Set up formatting for notebooks
Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun x writer -> fprintfn writer "%120A" x )
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

(** 

To move a tensor to the GPU use the following:

    let t2 = t.move(Device.GPU)
*)
