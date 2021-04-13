(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
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

Getting Started with DiffSharp and Torch
=========

To use the Torch backend for DiffSharp, reference one of

* [`DiffSharp-lite`](https://www.nuget.org/packages/DiffSharp-lite) - This includes the DiffSharp Torch backend but no LibTorch
  binaries.  You may need to add an explicit load of the relevant native library, e.g.

      open System.Runtime.InteropServices
      NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")

* [`DiffSharp-cpu`](https://www.nuget.org/packages/DiffSharp-cpu) - This includes the Torch backend using CPU only.

* [`DiffSharp-cuda-linux`](https://www.nuget.org/packages/DiffSharp-cuda-linux), [`DiffSharp-cuda-windows`](https://www.nuget.org/packages/DiffSharp-cuda-windows) - These include the Torch CPU/GPU backend for Linux and Windows respectively. Large download.

Then use:

    open DiffSharp

    dsharp.config(backend=Backend.Torch)

    let t = dsharp.tensor [ 0 .. 10 ]

Now examine the device and backend:

    let device = t.device
    let backend = t.backend


To move a tensor to the GPU use the following:

    let t2 = t.move(Device.GPU)

*)
printfn $"exiting {__SOURCE_FILE__}"
