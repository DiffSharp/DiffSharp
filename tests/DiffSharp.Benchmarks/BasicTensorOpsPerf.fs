// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

namespace DiffSharp.Benchmarks

open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs

open DiffSharp
open DiffSharp.Benchmarks

open System
open System.Threading
open DiffSharp.Backends
open TorchSharp
open TorchSharp.Tensor

/// For testing perf costs of the TorchSharp layer - going straght to the C++
module Ext =
    open System.Runtime.InteropServices
    [<DllImport("LibTorchSharp")>]
    extern IntPtr THSTorch_get_and_reset_last_err();

    [<DllImport("LibTorchSharp")>]
    extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

[<ShortRunJob>]
[<MarkdownExporterAttribute.GitHub; AsciiDocExporter; HtmlExporter; CsvExporter; RPlotExporter>]
[<GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)>]
[<CategoriesColumn; BaselineColumn>]
type BasicTensorOps() = 
    inherit BasicTensorTestMatrix()

    let mutable dtype = Unchecked.defaultof<Dtype>
    let mutable device = Unchecked.defaultof<Device>
    let mutable rawData = Unchecked.defaultof<Array>
    let mutable t = Unchecked.defaultof<Tensor>
    let mutable tvec = Unchecked.defaultof<Tensor>
    let mutable tmat = Unchecked.defaultof<Tensor>
    let mutable t0 = Unchecked.defaultof<Tensor>
    let mutable rawt = Unchecked.defaultof<RawTensor>
    let mutable rawtvec = Unchecked.defaultof<RawTensor>
    let mutable rawtmat = Unchecked.defaultof<RawTensor>
    let mutable rawt0 = Unchecked.defaultof<RawTensor>
    let mutable tt = Unchecked.defaultof<TorchTensor>
    let mutable ttvec = Unchecked.defaultof<TorchTensor>
    let mutable ttmat = Unchecked.defaultof<TorchTensor>
    let mutable tt0 = Unchecked.defaultof<TorchScalar>
    
    // store results temporarily to make sure nothing gets optimised away
    let mutable res = Unchecked.defaultof<Tensor>
    let mutable res3 = Unchecked.defaultof<_>
    let mutable res4 = Unchecked.defaultof<_>

    member perf.configure(backend, factor) = 
        match box tt with 
        | null -> 
            dtype <- (match perf.dtypeName with "int32" -> Dtype.Int32 | "float32" -> Dtype.Float32 | _ -> Dtype.Float64)
            device <- if perf.deviceName = "cpu" then Device.CPU else Device.GPU
            if not (dsharp.isDeviceTypeSupported(device.DeviceType, backend)) then failwith "device not supported"
            dsharp.config(dtype=dtype,backend=backend,device=device)
            rawData <- 
                match dtype with 
                | Dtype.Float32 -> Array.map float32 [| 1 .. perf.tensorSize |] :> Array
                | Dtype.Float64 -> Array.map double [| 1 .. perf.tensorSize |] :> Array
                | Dtype.Int32 -> Array.map int32 [| 1 .. perf.tensorSize |] :> Array
                | _ -> failwith "unknown dtype in perf suite"
            t <- dsharp.tensor [| 1 .. perf.tensorSize |]
            let matSize = int(sqrt(float perf.tensorSize))
            tvec <- dsharp.randint (1, 10, [| matSize |])
            tmat <- dsharp.randint (1, 10, [| matSize; matSize |])
            t0 <- dsharp.tensor 1.1
            rawt <- t.primalRaw
            rawtvec <- tvec.primalRaw
            rawtmat <- tmat.primalRaw
            rawt0 <- t0.primalRaw
            tt <- match rawt.Handle with :? TorchSharp.Tensor.TorchTensor as tt -> tt | _ -> Unchecked.defaultof<_>
            ttvec <- match rawtvec.Handle with :? TorchSharp.Tensor.TorchTensor as tt -> tt | _ -> Unchecked.defaultof<_>
            ttmat <- match rawtmat.Handle with :? TorchSharp.Tensor.TorchTensor as tt -> tt | _ -> Unchecked.defaultof<_>
            tt0 <- TorchSharp.TorchScalar.op_Implicit(1)
        | _ -> ()
        perf.numIterations(factor)

#if !TINY
    [<Benchmark(Baseline=true); BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(528) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(3459) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(535) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(3515) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(504) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(3502) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(29) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(52) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(29) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(57) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(22) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(45) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(30) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(32) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(30) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(32) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(24) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(25) else failwith "no time available" // PYTHON fromCpuData

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 2)
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> IntTensor.From(rawData :?> int32[])
                | Dtype.Int64 -> LongTensor.From(rawData :?> int64[])
                | Dtype.Float32 -> FloatTensor.From(rawData :?> single[])
                | Dtype.Float64 -> DoubleTensor.From(rawData :?> double[])
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 2)
        for _ in 1 .. n do res3 <- RawTensor.CreateFromFlatArray(rawData,  [| rawData.Length |])

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 2)
        for _ in 1 .. n do res  <- dsharp.tensor(rawData)

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 2)
        for _ in 1 .. n do res3 <- RawTensor.CreateFromFlatArray(rawData,  [| rawData.Length |])

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 2)
        for _ in 1 .. n do res  <- dsharp.tensor(rawData)
#endif

    //--------------------------------------------------------------
    // zeros

    [<Benchmark(Baseline=true); BenchmarkCategory("zeros")>]
    member perf.zeros_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(1645) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(5047) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(1652) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(5145) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(1616) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(5051) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(15) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(39) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(18) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(39) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(14) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(37) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(2) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(1) else failwith "no time available" // PYTHON zeros

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> IntTensor.Zeros([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Int64 -> LongTensor.Zeros([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Float32 -> DoubleTensor.Zeros([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Float64 -> FloatTensor.Zeros([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res3 <- RawTensor.Zeros(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res  <- dsharp.zeros( [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res3 <- RawTensor.Zeros(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res  <- dsharp.zeros( [| perf.tensorSize |])

#if !TINY
    //--------------------------------------------------------------
    // ones

    [<Benchmark(Baseline=true); BenchmarkCategory("ones")>]
    member perf.ones_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(1669) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(5078) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(1659) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(5033) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(1597) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(4971) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(14) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(38) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(16) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(39) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(14) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(40) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(2) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(1) else failwith "no time available" // PYTHON ones

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> IntTensor.Ones([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Int64 -> LongTensor.Ones([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Float32 -> DoubleTensor.Ones([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Float64 -> FloatTensor.Ones([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res3 <- RawTensor.Ones(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res  <- dsharp.ones( [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res3 <- RawTensor.Ones(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res  <- dsharp.ones( [| perf.tensorSize |])

    //--------------------------------------------------------------
    // rand

    [<Benchmark(Baseline=true); BenchmarkCategory("rand")>]
    member perf.rand_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(1981) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(5406) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(1971) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(5364) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(31) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(40) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(56) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(42) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(18) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(1) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(39) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(1) else failwith "no time available" // PYTHON rand

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> IntTensor.RandomIntegers(10L, [| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Int64 -> LongTensor.RandomIntegers(10L, [| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Float32 -> DoubleTensor.Random([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | Dtype.Float64 -> FloatTensor.Random([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- RawTensor.Random(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res  <- dsharp.rand( [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- RawTensor.Random(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res  <- dsharp.rand( [| perf.tensorSize |])

    //--------------------------------------------------------------
    // addition

    [<Benchmark(Baseline=true); BenchmarkCategory("addition")>]
    member perf.addition_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(742) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(3229) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(754) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(3145) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(714) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(3260) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(9) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(25) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(12) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(24) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(9) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(25) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(7) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(5) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(10) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(5) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(6) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(4) else failwith "no time available" // PYTHON addition

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do 
            res4 <- tt.Add(tt)

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res3 <- rawt.AddTT(rawt)

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res  <- t + t

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res3 <- rawt.AddTT(rawt)

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res  <- t + t


    //--------------------------------------------------------------
    // addScalar

    [<Benchmark(Baseline=true); BenchmarkCategory("addScalar")>]
    member perf.addScalar_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(1938) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(4212) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(1966) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(4186) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(1863) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(4333) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(19) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(33) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(23) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(32) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(18) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(32) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(7) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(5) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(10) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(5) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(7) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(4) else failwith "no time available" // PYTHON addScalar

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- tt.Add(tt0)

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- rawt.AddTT0(rawt0)

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res  <- t + t0

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- rawt.AddTT0(rawt0)

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res  <- t + t0

    //--------------------------------------------------------------
    // addWithAlpha

    [<Benchmark(Baseline=true); BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(530) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(1873) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(482) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(1782) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(485) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(1804) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(6) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(14) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(8) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(14) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(6) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(17) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(6) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(4) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(9) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(4) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(5) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(3) else failwith "no time available" // PYTHON addWithAlpha

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- tt.Add(tt, tt0)

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- rawt.AddTT(rawt.MulTT0(rawt0)) // TODO: no optimised routine in RawTensor as yet

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res  <- t.add(t.mul(t0)) // TODO: no optimised routine in Tensor as yet

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- rawt.AddTT(rawt.MulTT0(rawt0)) // TODO: no optimised routine in RawTensor as yet

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res  <- t.add(t.mul(t0)) // TODO: no optimised routine in Tensor as yet

    //--------------------------------------------------------------
    // addInPlace

    [<Benchmark(Baseline=true); BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_PyTorch() = 
        // This code gets injected, see Program.fs
        if perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(409) elif perf.tensorSize = 16 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(1643) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(385) elif perf.tensorSize = 16 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(1600) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(383) elif perf.tensorSize = 16 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(1613) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(5) elif perf.tensorSize = 2048 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(13) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(7) elif perf.tensorSize = 2048 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(13) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(5) elif perf.tensorSize = 2048 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(12) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cpu" then Thread.Sleep(6) elif perf.tensorSize = 65536 && perf.dtypeName = "float32" && perf.deviceName = "cuda" then Thread.Sleep(4) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cpu" then Thread.Sleep(8) elif perf.tensorSize = 65536 && perf.dtypeName = "float64" && perf.deviceName = "cuda" then Thread.Sleep(4) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cpu" then Thread.Sleep(5) elif perf.tensorSize = 65536 && perf.dtypeName = "int32" && perf.deviceName = "cuda" then Thread.Sleep(3) else failwith "no time available" // PYTHON addInPlace

    // TODO: add python here

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- tt.AddInPlace(tt)

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- rawt.AddTT(rawt) // TODO: no optimised routine in RawTensor as yet

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res  <- t + t // TODO: no optimised routine in RawTensor as yet

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- rawt.AddTT(rawt) // TODO: no optimised routine in RawTensor as yet

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res  <- t + t // TODO: no optimised routine in RawTensor as yet


    //--------------------------------------------------------------
    // matmul

    [<Benchmark(Baseline=true); BenchmarkCategory("matmul")>]
    member perf.matmul_PyTorch() : unit = 
        // This code gets injected, see Program.fs
        failwith "no time available yet" // PYTHON matmul
    // TODO: add python for this

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 1) 
        for _ in 1 .. n do 
            res4 <- ttmat.MatMul(ttmat)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 1) 
        for _ in 1 .. n do res3 <- rawtmat.MatMulTT(rawtmat)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 1) 
        for _ in 1 .. n do res  <- tmat.matmul(tmat)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 1) 
        for _ in 1 .. n do res3 <- rawtmat.MatMulTT(rawtmat)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 1) 
        for _ in 1 .. n do res  <- tmat.matmul(tmat)

#endif

    //[<Benchmark>]
    //member perf.sub_DiffSharp() = let n = perf.configure() in for _ in 1 .. n do res <- t + t

    //[<Benchmark>]
    //member perf.div() = let n = perf.configure() in for _ in 1 .. n do res <- t / t

    //[<Benchmark>]
    //member perf.sqrt() = let n = perf.configure() in for _ in 1 .. n do res <- sqrt(t)

    //[<Benchmark>]
    //member perf.relu() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.relu(t)

    //[<Benchmark>]
    //member perf.softmax() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.softmax(t, 0)

    //[<Benchmark>]
    //member perf.max() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.max(t)

    //[<Benchmark>]
    //member perf.sum() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.sum(t)

    //[<Benchmark>]
    //member perf.sin() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.sin(t)

    //[<Benchmark>]
    //member perf.lt() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.lt(t, t)

    //[<Benchmark>]
    //member perf.gradAddSum() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.grad (fun t -> (t + t).sum()) t

    //[<Benchmark>]
    //member perf.gradSinSum() = let n = perf.configure() in for _ in 1 .. n do res <- dsharp.grad (fun t -> (sin t).sum()) t


(*
[<ShortRunJob>]
type Training() = 

    [<Params (64, 256)>] 
    member val public n = 0 with get, set

    [<Params (100, 1000)>] 
    member val public din = 0 with get, set

    [<Params (10, 100)>] 
    member val public dout = 0 with get, set

    [<Benchmark>]
    member perf.trainSingleLinearLayer() =
        perf.configure()
        let n, din, dout = perf.n, perf.din, perf.dout
        let inputs  = dsharp.randn([n; din])
        let targets = dsharp.randn([n; dout])
        let dataset = TensorDataset(inputs, targets)
        let dataloader = dataset.loader(8, shuffle=true)

        // Trains a linear regressor
        let net = Linear(din, dout)
        let lr, mom, epochs = 1e-2, 0.9, 250
        let optimizer = SGD(net, lr=dsharp.tensor(lr), momentum=dsharp.tensor(mom), nesterov=true)
        for _ in 0..epochs do
            for _, inputs, targets in dataloader.epoch() do
                net.reverseDiff()
                let y = net.forward(inputs)
                let loss = dsharp.mseLoss(y, targets)
                loss.reverse()
                optimizer.step()
        let _y = net.forward inputs
        ()

*)


