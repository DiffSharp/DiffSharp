(*** condition: prepare ***)
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
// Google Colab only: uncomment and run the following to install dotnet and the F# kernel
// !bash <(curl -Ls https://raw.githubusercontent.com/gbaydin/scripts/main/colab_dotnet5.sh)
#endif // IPYNB
(*** condition: ipynb ***)
#if IPYNB
// Import DiffSharp package
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/{{fsdocs-source-basename}}.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# Installing

DiffSharp runs on [dotnet](https://dotnet.microsoft.com/), a cross-platform, open-source platform supported on Linux, macOS, and Windows.


## Packages

We provide several package bundles for a variety of use cases.

* [`DiffSharp-cpu`](https://www.nuget.org/packages/DiffSharp-cpu)</br>
  Includes LibTorch CPU binaries for Linux and Windows.
* [`DiffSharp-cuda-linux`](https://www.nuget.org/packages/DiffSharp-cuda-linux) / [`DiffSharp-cuda-windows`](https://www.nuget.org/packages/DiffSharp-cuda-windows)</br>
  Include LibTorch CPU and CUDA GPU binaries for Linux and Windows. Large download.
* [`DiffSharp-lite`](https://www.nuget.org/packages/DiffSharp-lite)</br>
  Includes the LibTorch backend but not the LibTorch binaries. 

### Using local LibTorch binaries

You can combine the `DiffSharp-lite` package bundle with existing local native binaries of LibTorch installed through other means (for example, by installing [PyTorch](https://pytorch.org/) using a Python package manager). If your GPU works in this PyTorch installation without any issues, it will also work in DiffSharp.

Before using the `Torch` backend in DiffSharp, you will have to add an explicit load of the LibTorch native library, which you can do as follows. In order to find the location of LibTorch binaries, searching for `libtorch.so` in your system might be helpful.

    open System.Runtime.InteropServices
    NativeLibrary.Load("/home/user/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


## Backends and Devices

DiffSharp currently provides two computation backends.

* The `Torch` backend is the default, recommended, backend based on [LibTorch](https://pytorch.org/cppdocs/), using the same C++ and CUDA implementations for tensor computations that power [PyTorch](https://pytorch.org/). On top of these raw tensors (LibTorch's ATen, excluding autograd), DiffSharp implements its own computation graph and differentiation capabilities. This backend requires platform-specific binaries of LibTorch, which we provide and test on Linux and Windows.

* The `Reference` backend is implemented purely in F# and can run on any hardware platform where dotnet can run. This backend has reasonable performance for use cases dominated by scalar operations, and is not recommended for use cases involving large tensor operations (such as machine learning). This backend is always available.

### Configuration

Selection of the backend is done using `cref:M:DiffSharp.dsharp.config`.

For example, the following selects the `Torch` backend with single precision tensors as the default tensor type and GPU (CUDA) execution.

*)

open DiffSharp

dsharp.config(dtype=Dtype.Float32, device=Device.GPU, backend=Backend.Torch)

(**
The following selects the `Reference` backend.
*)

dsharp.config(backend=Backend.Reference)

(**
A tensor's backend and device can be inspected as follows.

*)
let t = dsharp.tensor [ 0 .. 10 ]

let device = t.device
let backend = t.backend

(**
Tensors can be moved between devices (for example from CPU to GPU) using `cref:M:DiffSharp.Tensor.move(DiffSharp.Device)`. For example:
*)
let t2 = t.move(Device.GPU)

(**
## Using the DiffSharp Package

### Interactive Notebooks and Scripts

You can use DiffSharp in [dotnet interactive](https://github.com/dotnet/interactive) notebooks in [Visual Studio Code](https://code.visualstudio.com/) or [Jupyter](https://jupyter.org/), or in F# scripts (`.fsx` files), by referencing the package as follows:

    // Use one of the following three lines
    #r "nuget: DiffSharp-cpu" // Use the latest version
    #r "nuget: DiffSharp-cpu,*-*" // Use the latest pre-release version
    #r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}" // Use a specific version

    open DiffSharp

<img src="img/anim-intro-1.gif" width="85%" />

### Dotnet Applications

You can add DiffSharp to your dotnet application using the dotnet command-line interface (CLI).

    dotnet new console -lang "F#" -o src/app
    cd src/app
    dotnet add package --prerelease DiffSharp-cpu
    dotnet run
*)
