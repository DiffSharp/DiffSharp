<div align="left">
  <a href="https://diffsharp.github.io"> <img height="80px" src="docs/img/diffsharp-logo-text.png"></a>
</div>

-----------------------------------------

[Documentation](https://diffsharp.github.io/)

[![Build Status](https://travis-ci.org/DiffSharp/DiffSharp.svg?branch=dev)](https://travis-ci.org/DiffSharp/DiffSharp)
[![codecov](https://codecov.io/gh/DiffSharp/DiffSharp/branch/dev/graph/badge.svg)](https://codecov.io/gh/DiffSharp/DiffSharp)

This is the development branch of DiffSharp 1.0.

> **NOTE: This branch is undergoing development. It has incomplete code, functionality, and design that are likely to change without notice.**

## Getting Started

DiffSharp is normally used from an F# Jupyter notebook.  You can simply open examples directly in the browser, e.g.

* [index.ipynb](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=index.ipynb)

* [getting-started-torch.ipynb](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=getting-started-torch.ipynb)

To use locally you can install Jupyter and then:

    dotnet tool install -g --add-source "https://dotnet.myget.org/F/dotnet-try/api/v3/index.json" microsoft.dotnet-interactive
    dotnet interactive jupyter install

When using .NET Interactive it is best to completely turn off automatic HTML displays of outputs:

    Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
    Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )

You can also use DiffSharp from a script or an application.  Here are some example scripts with appropriate package references:

* [docs/index.fsx](http://diffsharp.github.io/index.fsx)

* [docs/getting-started-torch.fsx](http://diffsharp.github.io/getting-started-torch.fsx)

## Available packages and backends

Now reference an appropriate nuget package from https://nuget.org:

* [`DiffSharp-lite`](https://www.nuget.org/packages/DiffSharp-lite) - This is the reference backend.

* [`DiffSharp-cpu`](https://www.nuget.org/packages/DiffSharp-cpu) - This includes the Torch backend using CPU only.

* [`DiffSharp-cuda-linux`](https://www.nuget.org/packages/DiffSharp-cuda-linux), [`DiffSharp-cuda-windows`](https://www.nuget.org/packages/DiffSharp-cuda-windows) - These include the Torch CPU/GPU backend for Linux and Windows respectively. Large download.

For all but `DiffSharp-lite` add the following to your code:

    dsharp.config(backend=Backend.Torch)

## Using a pre-installed or self-built LibTorch 1.5.0

The Torch CPU and CUDA packages above are large.  If you already have `libtorch` 1.5.0 available on your machine you can

1. reference `DiffSharp-lite`

2. set `LD_LIBRARY_PATH` to include a directory containing the relevant `torch_cpu.so` and `torch_cuda.so`.

3. use `dsharp.config(backend=Backend.Torch)`

## Developing DiffSharp Libraries

To develop libraries built on DiffSharp, do the following:

1. reference `DiffSharp.Core` (and nothing else) in your library code.

2. reference `DiffSharp.Backends.Reference` in your correctness testing code.

3. reference `DiffSharp.Backends.Torch` and `libtorch-cpu` in your CPU testing code.

4. reference `DiffSharp.Backends.Torch` and `libtorch-cuda-linux` or `libtorch-cuda-windows` in your (optional) GPU testing code.
