(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.1"
#r "DiffSharp.Core.dll"
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
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/win-x64/native/torch_cpu.dll")
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX

(*** condition: ipynb ***)
#if IPYNB
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/win-x64/native/torch_cpu.dll")

// Set up formatting for notebooks
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
