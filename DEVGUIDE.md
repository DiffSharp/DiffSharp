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

2. Manually enable Torch CUDA binaries in `DiffSharp.Tests.fsproj` or set the `DIFFSHARP_TESTGPU` environment variable to `true` (e.g. `dotnet test /p:DIFFSHARP_TESTGPU=true`)

3. Verify that `dsharp.isCudaEnabled()` is returning true and GPU testing is enabled in `TestUtil.fs`.



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
frame #4: c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, at::Tensor const&, at::Tensor const&), &torch::autograd::VariableType::(anonymous namespace)::mm(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, at::Tensor const&>>, at::Tensor (c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&) + 1228 (0x301c7c0a0 in libtorch_cpu.dylib)
frame #5: at::_ops::mm::call(at::Tensor const&, at::Tensor const&) + 192 (0x300af0478 in libtorch_cpu.dylib)
frame #6: at::native::_matmul_impl(at::Tensor&, at::Tensor const&, at::Tensor const&) + 692 (0x3003201ec in libtorch_cpu.dylib)
frame #7: at::native::matmul(at::Tensor const&, at::Tensor const&) + 76 (0x3003210cc in libtorch_cpu.dylib)
frame #8: at::_ops::matmul::call(at::Tensor const&, at::Tensor const&) + 192 (0x300bdc5cc in libtorch_cpu.dylib)
frame #9: THSTensor_matmul + 52 (0x1361c65c0 in libLibTorchSharp.dylib)
frame #10: 0x0 + 10806075020 (0x284179e8c in ???)
frame #11: 0x0 + 10801407760 (0x283d06710 in ???)
frame #12: 0x0 + 10796142560 (0x283800fe0 in ???)
frame #13: 0x0 + 10796133464 (0x2837fec58 in ???)
frame #14: 0x0 + 10796113888 (0x2837f9fe0 in ???)
frame #15: 0x0 + 10807490084 (0x2842d3624 in ???)
frame #16: CallDescrWorkerInternal + 132 (0x105282408 in libcoreclr.dylib)
frame #17: CallDescrWorkerWithHandler(CallDescrData*, int) + 116 (0x1050f3384 in libcoreclr.dylib)
frame #18: CallDescrWorkerReflectionWrapper(CallDescrData*, Frame*) + 112 (0x10519f440 in libcoreclr.dylib)
frame #19: RuntimeMethodHandle::InvokeMethod(Object*, Span<Object*>*, SignatureNative*, bool, bool) + 2320 (0x1051a0028 in libcoreclr.dylib)
frame #20: 0x0 + 10794138572 (0x283617bcc in ???)
frame #21: 0x0 + 10788166936 (0x283065d18 in ???)
frame #22: 0x0 + 10788166736 (0x283065c50 in ???)
frame #23: 0x0 + 10788166572 (0x283065bac in ???)
frame #24: 0x0 + 10788166296 (0x283065a98 in ???)
frame #25: 0x0 + 10788165652 (0x283065814 in ???)
frame #26: 0x0 + 10788165536 (0x2830657a0 in ???)
frame #27: 0x0 + 10788165372 (0x2830656fc in ???)
frame #28: 0x0 + 10794179444 (0x283621b74 in ???)
frame #29: 0x0 + 10788165060 (0x2830655c4 in ???)
frame #30: 0x0 + 10788164680 (0x283065448 in ???)
frame #31: 0x0 + 10788158484 (0x283063c14 in ???)
frame #32: 0x0 + 10788128032 (0x28305c520 in ???)
frame #33: 0x0 + 10788127328 (0x28305c260 in ???)
frame #34: 0x0 + 10788116464 (0x2830597f0 in ???)
frame #35: 0x0 + 10788148552 (0x283061548 in ???)
frame #36: 0x0 + 10788146364 (0x283060cbc in ???)
frame #37: 0x0 + 10788136876 (0x28305e7ac in ???)
frame #38: 0x0 + 10788128032 (0x28305c520 in ???)
frame #39: 0x0 + 10788127328 (0x28305c260 in ???)
frame #40: 0x0 + 10788123112 (0x28305b1e8 in ???)
frame #41: 0x0 + 10741825676 (0x28043408c in ???)
frame #42: CallDescrWorkerInternal + 132 (0x105282408 in libcoreclr.dylib)
frame #43: DispatchCallSimple(unsigned long*, unsigned int, unsigned long, unsigned int) + 272 (0x1050f36c0 in libcoreclr.dylib)
frame #44: ThreadNative::KickOffThread_Worker(void*) + 148 (0x105108778 in libcoreclr.dylib)
frame #45: ManagedThreadBase_DispatchOuter(ManagedThreadCallState*) + 260 (0x1050be74c in libcoreclr.dylib)
frame #46: ManagedThreadBase::KickOff(void (*)(void*), void*) + 32 (0x1050becb8 in libcoreclr.dylib)
frame #47: ThreadNative::KickOffThread(void*) + 172 (0x105108850 in libcoreclr.dylib)
frame #48: CorUnix::CPalThread::ThreadEntry(void*) + 380 (0x104fd20e4 in libcoreclr.dylib)
frame #49: _pthread_start + 148 (0x191bc7fa8 in libsystem_pthread.dylib)
frame #50: thread_start + 8 (0x191bc2da0 in libsystem_pthread.dylib)
```