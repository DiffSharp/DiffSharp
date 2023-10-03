# DiffSharp - Development Guide

You can clone this repository to your machine as follows:
```
git clone --branch dev https://github.com/DiffSharp/DiffSharp.git
cd DiffSharp
```

## Run tests

Required:
- Install [.NET Core SDK](https://dotnet.microsoft.com/download) for your system

Use the following command in the root directory of this repository:
```
dotnet test
```

## Build DiffSharp in Docker

Required:
- Install [Docker](https://hub.docker.com/search/?type=edition&offering=community) for your system

Build a Docker image called `diffsharp`. This will work without any local .NET Core installation and build DiffSharp inside the image.
```
docker build -t diffsharp .
```

Use the following to instantiate a Docker container from the `diffsharp` image and run the tests inside:
```
docker run --rm diffsharp dotnet test
```

## Building against locally built TorchSharp packages

To add features you may have extend TorchSharp to make extra features of LibTorch available.

The build is set up to look for a parallel build of TorchSharp, e.g.

    C:\GitHub\dsyme\DiffSharp
    C:\GitHub\dsyme\TorchSharp

To build, test and pack TorchSharp in that repo do this:

    .\build build
    .\build test
    .\build pack

You will see something like this

    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg'.
    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/LibTorch.Redist.0.3.0-local-Debug-20200520.nupkg'.

with warning:

    warning : Packages will be incomplete and unusable on other platforms...

To consume the packages into DiffSharp adjust TorchSharpVersion in Directory.Build.props.

When rebuilding the TorchSharp you will need to clear your package cache to pick up the new nuget package with the same version id, e.g.

    rmdir /q /s %USERPROFILE%\.nuget\packages\torchsharp
    rmdir /q /s %USERPROFILE%\.nuget\packages\LibTorch.Redist
    dotnet restore

The LibTorch packages are quite large and you may need to watch disk space.

## The Reference Backend

The "Reference" backend defines the semantics we expect of the Torch backend.

Sometimes configurations of Torch expose small differences in semantics (e.g. when using CUDA, or functionality not suppored for integer tensors).  We generally seek to paper
over those cracks by working around the problems in the Torch backend. 

## Developing and Testing on GPU

By default in-branch testing is only done on CPU.  To enable on GPU/CUDA you must:

1. Make sure you have a device eligible for CUDA 11.1 and all device drivers installed (e.g. install the appropriate NVIDIA CUDA SDK)

2. Use `dotnet test /p:DIFFSHARP_TESTGPU=true`

3. Verify that `dsharp.isCudaEnabled(Device.GPU)` is returning true and GPU testing is enabled in `TestUtil.fs`.

## Micro Performance Benchmarking 

Python numbers must be collected in a separate run, they are currently injected back into source code (ugh)
to get figures in one report.  There are better ways to do this.

To update Python benchmarks on your machine (note, writes back results into source code)

    dotnet run --project tests\DiffSharp.Benchmarks.Python\DiffSharp.Benchmarks.Python.fsproj -c Release --filter "*"

This takes a while to run.

To run benchmarks:

    dotnet run --project tests\DiffSharp.Benchmarks\DiffSharp.Benchmarks.fsproj -c Release --filter "*"

To filter etc., see `--help`

## TorchSharp backend on macos arm64

In order to use TorchSharp backend on macOs arm64 platform:

* you need to build TorchSharp from this PR: https://github.com/dotnet/TorchSharp/pull/903
* you need to adjust according to "Building against locally built TorchSharp packages" section of this document

At the time of this writing, there is one failing test and 327/328 passing:

```
The active test run was aborted. Reason: Test host process crashed : libc++abi: terminating due to uncaught exception of type c10::Error: "addmm_impl_cpu_" not implemented for 'Half'
Exception raised from operator() at /tmp/pytorch-20230625-11028-1u2efwj/aten/src/ATen/native/LinearAlgebra.cpp:1433 (most recent call first):
frame #0: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 92 (0x104c861c0 in libc10.dylib)
frame #1: at::native::addmm_impl_cpu_(at::Tensor&, at::Tensor const&, at::Tensor, at::Tensor, c10::Scalar const&, c10::Scalar const&) + 4484 (0x3003161f4 in libtorch_cpu.dylib)
frame #2: at::native::structured_mm_out_cpu::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&) + 184 (0x300316704 in libtorch_cpu.dylib)
frame #3: at::(anonymous namespace)::wrapper_CPU_mm(at::Tensor const&, at::Tensor const&) + 96 (0x300d0c7f8 in libtorch_cpu.dylib)
...
```
