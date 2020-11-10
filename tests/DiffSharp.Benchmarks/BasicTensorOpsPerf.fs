// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

namespace DiffSharp.Benchmarks.BasicTensorOps

open System
open DiffSharp
open DiffSharp.Data
open DiffSharp.Model
open DiffSharp.Optim
open TorchSharp
open TorchSharp.Tensor
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Columns
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Running
open BenchmarkDotNet.Order
open DiffSharp.Backends

/// For testing perf costs of the TorchSharp layer - going straght to the C++
module Ext =
    open System.Runtime.InteropServices
    [<DllImport("LibTorchSharp")>]
    extern IntPtr THSTorch_get_and_reset_last_err();

    [<DllImport("LibTorchSharp")>]
    extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

[<AutoOpen>]
module Helpers =
    [<Literal>]
    let Layer_TorchSharp = "1 - TorchSharp"
    [<Literal>]
    let Layer_RawTensor = "2 - RawTensor"
    [<Literal>]
    let Layer_Tensor = "3 - Tensor"

[<ShortRunJob>]
[<MarkdownExporterAttribute.GitHub; AsciiDocExporter; HtmlExporter; CsvExporter; RPlotExporter>]
[<GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByMethod)>]
type BasicTensorOps() = 

    let mutable dtype = Unchecked.defaultof<Dtype>
    let mutable device = Unchecked.defaultof<Device>
    let mutable backend = Unchecked.defaultof<Backend>
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
    let N = pown 2 18

    member perf.configure() = 
        match box tt with 
        | null -> 
            dtype <- (match perf.dtypeName with "int32" -> Dtype.Int32 | "float32" -> Dtype.Float32 | _ -> Dtype.Float64)
            backend <- if perf.backendName = "torch" then Backend.Torch else Backend.Reference
            device <- if perf.deviceName = "cpu" then Device.CPU else Device.GPU
            if not (dsharp.isDeviceTypeSupported(device.DeviceType, backend)) then failwith "device not supported"
            dsharp.config(dtype=dtype,backend=backend,device=device)
            rawData <- Array.map float32 [| 1 .. perf.tensorSize |]
            t <- dsharp.tensor [| 1 .. perf.tensorSize |]
            let matSize = int(sqrt(float perf.tensorSize))
            tvec <- dsharp.rand [| matSize |]
            tmat <- dsharp.rand [| matSize; matSize |]
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
        N/perf.tensorSize

    [<Params (16, 2048, 65536)>] 
    member val public tensorSize = 0 with get, set

    //[<Params ("float32")>] 
    [<Params ("int32", "float32", "float64")>] 
    member val public dtypeName = "" with get, set

    //[<Params ("cpu")>] 
    [<Params ("cpu", "gpu")>] 
    member val public deviceName = "" with get, set

    [<Params ("torch", "reference")>] 
    member val public backendName = "" with get, set

    [<Params (Layer_TorchSharp, Layer_RawTensor, "3 - Tensor")>] 
    member val public tensorLayer = "" with get, set

    [<Benchmark>]
    member perf.fromCpuData() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do 
                    res4 <- 
                        match dtype with 
                        | Dtype.Int32 -> IntTensor.From(rawData :?> int32[])
                        | Dtype.Int64 -> LongTensor.From(rawData :?> int64[])
                        | Dtype.Float32 -> FloatTensor.From(rawData :?> single[])
                        | Dtype.Float64 -> DoubleTensor.From(rawData :?> double[])
                        | _ -> failwith "unknown dtype in perf testing"
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- RawTensor.CreateFromFlatArray(rawData,  [| rawData.Length |])
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res  <- dsharp.tensor(rawData)

    [<Benchmark>]
    member perf.zeros() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do res4 <- FloatTensor.Zeros([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- RawTensor.Zeros(Shape.create [| perf.tensorSize |])
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res  <- dsharp.zeros( [| perf.tensorSize |])

    [<Benchmark>]
    member perf.ones() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do res4 <- FloatTensor.Ones([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- RawTensor.Ones(Shape.create [| perf.tensorSize |])
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res  <- dsharp.ones( [| perf.tensorSize |])

    [<Benchmark>]
    member perf.rand() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do res4 <- FloatTensor.Random([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- RawTensor.Random(Shape.create [| perf.tensorSize |])
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res  <- dsharp.rand( [| perf.tensorSize |])


    [<Benchmark>]
    member perf.randn() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do res4 <- FloatTensor.RandomN([| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- RawTensor.RandomNormal(Shape.create [| perf.tensorSize |])
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res  <- dsharp.randn([| perf.tensorSize |])

    [<Benchmark>]
    member perf.randint() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do res4 <- FloatTensor.RandomIntegers(10L, [| int64 perf.tensorSize |] , enum (int Device.Default.DeviceType), Device.Default.DeviceIndex)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- RawTensor.RandomInt(Shape.create [| perf.tensorSize |], 0, 10)
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res  <- dsharp.randint(0, 10, [| perf.tensorSize |])

    [<Benchmark>]
    member perf.matmul() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
                for _ in 1 .. n do res4 <- ttmat.MatMul(ttmat)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawtmat.MatMulTT(rawtmat)
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- tmat.matmul(tmat)

    // // Testing the cost of the error checks in TorchSharp
    // //
    // [<Benchmark>]
    //  This can be significant for tensor sizes < approx 16
    // member perf.TorchSharp_CheckForErrors() =
    //     let n = perf.configure()
    //     for _ in 1 .. n do 
    //         res2 <- Ext.THSTorch_get_and_reset_last_err()

    [<Benchmark>]
    member perf.addition() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
               for _ in 1 .. n do res4 <- tt.Add(tt)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.AddTT(rawt)
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t + t

    [<Benchmark>]
    member perf.addScalar() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then  
               for _ in 1 .. n do res4 <- tt.Add(tt0)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.AddTT0(rawt0)
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t + t0

    [<Benchmark>]
    member perf.addWithAlpha() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
              for _ in 1 .. n do res4 <- tt.Add(tt, tt0)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.AddTT(rawt.MulTT0(rawt0))
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t.add(t0.mul(t0))

    [<Benchmark>]
    member perf.addWithAlphaInPlace() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
               for _ in 1 .. n do res4 <- tt.AddInPlace(tt, tt0)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.AddTT(rawt.MulTT0(rawt0)) // NO CORRESPONDING INPLACE YET 
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t.add(t0.mul(t0)) // NO CORRESPONDING INPLACE YET 

    [<Benchmark>]
    member perf.addScalarWithAlpha() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
              for _ in 1 .. n do res4 <- tt.Add(tt0, tt0)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.AddTT0(rawt0.MulTT0(rawt0))
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t.add(t0.mul(t0))

    [<Benchmark>]
    member perf.addScalarWithAlphaInPlace() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
              for _ in 1 .. n do res4 <- tt.AddInPlace(tt0, tt0)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.AddTT0(rawt0.MulTT0(rawt0))
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t.add(t0.mul(t0))

    [<Benchmark>]
    member perf.mul() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
               for _ in 1 .. n do res4 <- tt.Mul(tt)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.MulTT(rawt)
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t + t

    [<Benchmark>]
    member perf.mulInPlace() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
               for _ in 1 .. n do res4 <- tt.MulInPlace(tt)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.MulTT(rawt) // NO CORRESPONDING INPLACE YET 
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t * t // NO CORRESPONDING INPLACE YET 

    [<Benchmark>]
    member perf.mulScalar() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
               for _ in 1 .. n do res4 <- tt.Mul(tt)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.MulTT(rawt)
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t + t

    [<Benchmark>]
    member perf.mulScalarInPlace() = 
        let n = perf.configure() 
        match perf.tensorLayer with 
        | Layer_TorchSharp -> 
            if backend = Backend.Torch then 
               for _ in 1 .. n do res4 <- tt.MulInPlace(tt0)
            else failwith "n/a"
        | Layer_RawTensor -> 
            for _ in 1 .. n do res3 <- rawt.MulTT0(rawt) // NO CORRESPONDING INPLACE YET 
        | _ (* "3 - Tensor" *) -> 
            for _ in 1 .. n do res <- t * t0 // NO CORRESPONDING INPLACE YET 

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

[<ShortRunJob>]
type Training() = 

    member perf.configure() = 
        let dtype = (match perf.dtype with "int32" -> Dtype.Int32 | "float32" -> Dtype.Float32 | _ -> Dtype.Float64)
        let backend = if perf.backend = "torch" then Backend.Torch else Backend.Reference
        let device = if perf.device = "cpu" then Device.CPU else Device.GPU
        if not (dsharp.isDeviceTypeSupported(device.DeviceType, backend)) then failwith "device not supported"
        dsharp.config(dtype=dtype,backend=backend,device=device)

    [<Params ("float32")>] 
    //[<Params ("float32", "float64")>] 
    member val public dtype = "" with get, set

    [<Params ("cpu")>] //[<Params ("cpu", "gpu")>] 
    member val public device = "" with get, set

    [<Params ("torch", "reference")>] 
    member val public backend = "" with get, set

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

