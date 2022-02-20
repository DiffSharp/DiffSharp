(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net6.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.Torch.dll"
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
DiffSharp.dsharp.config(backend=DiffSharp.Backend.Reference)
DiffSharp.dsharp.seed(123)

(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
// Google Colab only: uncomment and run the following to install dotnet and the F# kernel
// !bash <(curl -Ls https://raw.githubusercontent.com/gbaydin/scripts/main/colab_dotnet6.sh)
#endif // IPYNB
(*** condition: ipynb ***)
#if IPYNB
// Import DiffSharp package
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypesFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/{{fsdocs-source-basename}}.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# Installing

DiffSharp runs on [dotnet](https://dotnet.microsoft.com/), a cross-platform, open-source platform supported on Linux, macOS, and Windows.

There are various ways in which you can run DiffSharp, the main ones being: [interactive notebooks](https://github.com/dotnet/interactive) supporting [Visual Studio Code](https://code.visualstudio.com/) and [Jupyter](https://jupyter.org/); running in a [REPL](https://github.com/jonsequitur/dotnet-repl); running [script files](https://docs.microsoft.com/en-us/dotnet/fsharp/tools/fsharp-interactive/); and [compiling, packing, and publishing](https://docs.microsoft.com/en-us/dotnet/core/introduction) performant binaries.


## Interactive Notebooks and Scripts

You can use DiffSharp in [dotnet interactive](https://github.com/dotnet/interactive) notebooks in [Visual Studio Code](https://code.visualstudio.com/) or [Jupyter](https://jupyter.org/), or in F# scripts (`.fsx` files), by referencing the package as follows:

    // Use one of the following three lines
    #r "nuget: DiffSharp-cpu" // Use the latest version
    #r "nuget: DiffSharp-cpu, *-*" // Use the latest pre-release version
    #r "nuget: DiffSharp-cpu, 1.0.1" // Use a specific version

    open DiffSharp

</br>
<img src="img/anim-intro-1.gif" width="85%" />

## Dotnet Applications

You can add DiffSharp to your dotnet application using the [dotnet](https://dotnet.microsoft.com/) command-line interface (CLI).

For example, the following creates a new F# console application and adds the latest pre-release version of the `DiffSharp-cpu` package as a dependency.

    dotnet new console -lang "F#" -o src/app
    cd src/app
    dotnet add package --prerelease DiffSharp-cpu
    dotnet run

## Packages

We provide several package bundles for a variety of use cases.

* [DiffSharp-cpu](https://www.nuget.org/packages/DiffSharp-cpu)</br>
  Includes LibTorch CPU binaries for Linux, macOS, and Windows.
* [DiffSharp-cuda-linux](https://www.nuget.org/packages/DiffSharp-cuda-linux) / [DiffSharp-cuda-windows](https://www.nuget.org/packages/DiffSharp-cuda-windows)</br>
  Include LibTorch CPU and CUDA GPU binaries for Linux and Windows. Large download.
* [DiffSharp-lite](https://www.nuget.org/packages/DiffSharp-lite)</br>
  Includes the Torch backend but not the LibTorch binaries. 

### Using local LibTorch binaries (optional)

You can combine the `DiffSharp-lite` package bundle with existing local native binaries of LibTorch for your OS (Linux, Mac, or Windows) installed through other means. 

LibTorch is the main tensor computation core implemented in C++/CUDA and it is used by PyTorch in Python and by other projects in various programming languages. The following are two common ways of having LibTorch in your system.

* If you use Python and have [PyTorch](https://pytorch.org/) installed, this comes with LibTorch as a part of the PyTorch distribution. If your GPU works in this PyTorch installation without any issues, it will also work in DiffSharp.
* You can download the native LibTorch package without Python by following the [get started](https://pytorch.org/get-started/locally/) instructions in the PyTorch website, and extracting the downloaded archive to a folder in your system.

Before using the `Torch` backend in DiffSharp, you will have to add an explicit load of the LibTorch native library, which you can do as follows. In order to find the location of LibTorch binaries, searching for `libtorch.so` in your system might be helpful. Note that this file is called `libtorch.so` in Linux, `libtorch.dylib` in macOS, and `torch.dll` in Windows.

    open System.Runtime.InteropServices
    NativeLibrary.Load("/home/user/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


## Backends and Devices

DiffSharp currently provides two computation backends.

* The `Torch` backend is the default and recommended backend based on [LibTorch](https://pytorch.org/cppdocs/), using the same C++ and CUDA implementations for tensor computations that power [PyTorch](https://pytorch.org/). On top of these raw tensors (LibTorch's ATen, excluding autograd), DiffSharp implements its own computation graph and differentiation capabilities. This backend requires platform-specific binaries of LibTorch, which we provide and test on Linux, macOS, and Windows.

* The `Reference` backend is implemented purely in F# and can run on any hardware platform where [dotnet](https://dotnet.microsoft.com/) can run (for example iOS, Android, Raspberry Pi). This backend has reasonable performance for use cases dominated by scalar and small tensor operations, and is not recommended for use cases involving large tensor operations (such as machine learning). This backend is always available.

### Configuration of Default Backend, Device, and Tensor Type

Selection of the default backend, device, and tensor type is done using `cref:M:DiffSharp.dsharp.config`.

* `cref:T:DiffSharp.Dtype` choices available: `BFloat16`, `Bool`, `Byte`, `Float16`, `Float32`, `Float64`, `Int16`, `Int32`, `Int64`, `Int8`

* `cref:T:DiffSharp.Device` choices available: `CPU`, `GPU`

* `cref:T:DiffSharp.Backend` choices available: `Reference`, `Torch`

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
## Developing DiffSharp Libraries

To develop libraries built on DiffSharp, you can use the following guideline to reference the various packages.

* Reference `DiffSharp.Core` and `DiffSharp.Data` in your library code.
* Reference `DiffSharp.Backends.Reference` in your correctness testing code.
* Reference `DiffSharp.Backends.Torch` and `libtorch-cpu` in your CPU testing code.
* Reference `DiffSharp.Backends.Torch` and `libtorch-cuda-linux` or `libtorch-cuda-windows` in your (optional) GPU testing code.

*)
