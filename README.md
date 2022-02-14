<div align="left">
  <a href="https://diffsharp.github.io"> <img height="80px" src="docs/img/diffsharp-logo-text.png"></a>
</div>

-----------------------------------------

[Documentation](https://diffsharp.github.io/)

[![Build Status](https://github.com/DiffSharp/DiffSharp/workflows/Build/test/docs/publish/badge.svg)](https://github.com/DiffSharp/DiffSharp/actions)
[![Coverage Status](https://coveralls.io/repos/github/DiffSharp/DiffSharp/badge.svg?branch=)](https://coveralls.io/github/DiffSharp/DiffSharp?branch=)

This is the development branch of DiffSharp 1.0.

> **NOTE: This branch is undergoing development. It has incomplete code, functionality, and design that are likely to change without notice.**

## Getting Started

DiffSharp is normally used from an F# Jupyter notebook.  You can simply open examples directly in the browser, e.g.

* [index.ipynb](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=index.ipynb)

* [getting-started-install.ipynb](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=getting-started-install.ipynb)

To use locally in [Visual Studio Code](https://code.visualstudio.com/):

- Install [.NET Interactive Notebooks for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode)

- After opening an `.ipynb` execute `Ctrl-Shift-P` for the command palette and chose `Reopen Editor With...` then `.NET Interactive Notebooks`

- To restart the kernel use `restart` from the command palette.

To use locally in Jupyter, first install Jupyter and then:

    dotnet tool install -g microsoft.dotnet-interactive
    dotnet interactive jupyter install

When using .NET Interactive it is best to completely turn off automatic HTML displays of outputs:

    Formatter.SetPreferredMimeTypesFor(typeof<obj>, "text/plain")
    Formatter.Register(fun x writer -> fprintfn writer "%120A" x )

You can also use DiffSharp from a script or an application.  Here are some example scripts with appropriate package references:

* [docs/index.fsx](http://diffsharp.github.io/index.fsx)

* [docs/getting-started-install.fsx](http://diffsharp.github.io/getting-started-install.fsx)

## Available packages and backends

Now reference an appropriate nuget package from https://nuget.org:

* [`DiffSharp-lite`](https://www.nuget.org/packages/DiffSharp-lite) - This is the reference backend.

* [`DiffSharp-cpu`](https://www.nuget.org/packages/DiffSharp-cpu) - This includes the Torch backend using CPU only.

* [`DiffSharp-cuda-linux`](https://www.nuget.org/packages/DiffSharp-cuda-linux) - This includes the Torch CPU/CUDA 11.1 backend for Linux. Large download. Requires .NET 6 SDK, version `6.0.100-preview.5.21302.13` or greater.

* [`DiffSharp-cuda-windows`](https://www.nuget.org/packages/DiffSharp-cuda-windows) - This includes the Torch CPU/CUDA 11.1 backend for Windows. Large download.

For all but `DiffSharp-lite` add the following to your code:

    dsharp.config(backend=Backend.Torch)

## Using a pre-installed or self-built LibTorch 1.8.0

The Torch CPU and CUDA packages above are large.  If you already have `libtorch` 1.8.0 available on your machine you can

1. reference `DiffSharp-lite`

2. set `LD_LIBRARY_PATH` to include a directory containing the relevant `torch.so`, `torch_cpu.so` and `torch_cuda.so`, or
   execute [NativeLibrary.Load](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativelibrary.load?view=net-5.0) on
   `torch.so`.

3. use `dsharp.config(backend=Backend.Torch)`

## Developing DiffSharp Libraries

To develop libraries built on DiffSharp, do the following:

1. reference `DiffSharp.Core` and `DiffSharp.Data` in your library code.

2. reference `DiffSharp.Backends.Reference` in your correctness testing code.

3. reference `DiffSharp.Backends.Torch` and `libtorch-cpu` in your CPU testing code.

4. reference `DiffSharp.Backends.Torch` and `libtorch-cuda-linux` or `libtorch-cuda-windows` in your (optional) GPU testing code.
