// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Benchmarks

open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs

open DiffSharp
open DiffSharp.Benchmarks

open System
open System.IO
open System.Threading
open DiffSharp.Backends
open TorchSharp

type TorchDevice = Torch.Device
type Device = DiffSharp.Device

[<AutoOpen>]
module Extensions =
    type DiffSharp.DeviceType with 
        member x.ToTorch : TorchSharp.DeviceType = enum (int x)

    type DiffSharp.Device with 
        member x.ToTorch = torch.Device(x.DeviceType.ToTorch, x.DeviceIndex)


/// For testing perf costs of the TorchSharp layer - going straght to the C++
module Ext =
    open System.Runtime.InteropServices
    [<DllImport("LibTorchSharp")>]
    extern IntPtr THSTorch_get_and_reset_last_err();

    [<DllImport("LibTorchSharp")>]
    extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

module PythonResults =
    let pythonResults = 
        let pyFile = Path.Combine(__SOURCE_DIRECTORY__, "..", "DiffSharp.Benchmarks.Python", "results.csv")
        if File.Exists(pyFile) then  
            let lines = File.ReadAllLines(pyFile)
            dict [ for line in lines do 
                       let c = line.LastIndexOf("," )
                       if c <> -1 then 
                           let res = line[0..c-1], int line[c+1..] 
                           printfn "%A" res
                           res]
        else 
            printfn "*** No python results found at '%s', have you run DiffSharp.Benchmarks.Python?" pyFile
            dict [ ]

[<ShortRunJob>]
[<MarkdownExporterAttribute.GitHub; AsciiDocExporter; HtmlExporter; CsvExporter; RPlotExporter>]
[<GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)>]
[<CategoriesColumn; BaselineColumn>]
type BasicTensorOps() = 
    inherit BasicTensorTestMatrix()

    let mutable dtype = Unchecked.defaultof<Dtype>
    let mutable device = Unchecked.defaultof<Device>
    let mutable rawData = Unchecked.defaultof<Array>
    let mutable dsharpTensor = Unchecked.defaultof<Tensor>
    let mutable dsharpMatrixTensor = Unchecked.defaultof<Tensor>
    let mutable dsharpScalarTensor = Unchecked.defaultof<Tensor>
    let mutable dsharpScalar = Unchecked.defaultof<scalar>
    let mutable rawTensor = Unchecked.defaultof<RawTensor>
    let mutable rawMatrixTensor = Unchecked.defaultof<RawTensor>
    let mutable rawScalarTensor = Unchecked.defaultof<RawTensor>
    let mutable rawScalar = Unchecked.defaultof<scalar>
    let mutable torchTensor = Unchecked.defaultof<torch.Tensor>
    let mutable matrixTorchTensor = Unchecked.defaultof<torch.Tensor>
    let mutable torchScalar = Unchecked.defaultof<TorchSharp.Scalar>
    
    // store results temporarily to make sure nothing gets optimised away
    let mutable res = Unchecked.defaultof<Tensor>
    let mutable res3 = Unchecked.defaultof<_>
    let mutable res4 = Unchecked.defaultof<_>

    member perf.simulatePythonResult(nm) =
        // Note, this string allocation and dictionary lookup can affect result
        let key = nm + string perf.tensorSize + perf.dtypeName + perf.deviceName
        if PythonResults.pythonResults.ContainsKey(key) then
            let time = PythonResults.pythonResults[key]
            Thread.Sleep(time)
        else  
            failwithf "key '%s' not found in python results, have you run DiffSharp.Benchmarks.Python?" key

    member perf.configure(backend, factor) = 
        match box torchTensor with 
        | null -> 
            dtype <- (match perf.dtypeName with "int32" -> Dtype.Int32 | "float32" -> Dtype.Float32 | _ -> Dtype.Float64)
            device <- if perf.deviceName = "cpu" then Device.CPU else Device.GPU
            if not (dsharp.isDeviceTypeAvailable(device.DeviceType, backend)) then failwith "device not supported"
            dsharp.config(dtype=dtype,backend=backend,device=device)
            rawData <- 
                match dtype with 
                | Dtype.Float32 -> Array.map float32 [| 1 .. perf.tensorSize |] :> Array
                | Dtype.Float64 -> Array.map double [| 1 .. perf.tensorSize |] :> Array
                | Dtype.Int32 -> Array.map int32 [| 1 .. perf.tensorSize |] :> Array
                | _ -> failwith "unknown dtype in perf suite"
            dsharpTensor <- dsharp.tensor [| 1 .. perf.tensorSize |]
            let matSize = int(sqrt(float perf.tensorSize))
            dsharpMatrixTensor <- dsharp.randint (1, 10, [| matSize; matSize |])
            dsharpScalarTensor <- dsharp.tensor 1.1
            dsharpScalar <- 3
            rawTensor <- dsharpTensor.primalRaw
            rawMatrixTensor <- dsharpMatrixTensor.primalRaw
            rawScalarTensor <- dsharpScalarTensor.primalRaw
            rawScalar <- 3
            torchTensor <- match rawTensor.Handle with :? torch.Tensor as tt -> tt | _ -> Unchecked.defaultof<_>
            matrixTorchTensor <- match rawMatrixTensor.Handle with :? torch.Tensor as tt -> tt | _ -> Unchecked.defaultof<_>
            torchScalar <- TorchSharp.Scalar.op_Implicit(3)
        | _ -> ()
        perf.numIterations(factor)

    [<Benchmark(Baseline=true); BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_PyTorch() = perf.simulatePythonResult("fromCpuData")

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 2)
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> torch.tensor(rawData :?> int32[])
                | Dtype.Int64 -> torch.tensor(rawData :?> int64[])
                | Dtype.Float32 -> torch.tensor(rawData :?> single[])
                | Dtype.Float64 -> torch.tensor(rawData :?> double[])
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 2)
        for _ in 1 .. n do res3 <- RawTensor.CreateFromFlatArray(rawData,  [| rawData.Length |])

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 2)
        for _ in 1 .. n do res <- dsharp.tensor(rawData)

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 2)
        for _ in 1 .. n do res3 <- RawTensor.CreateFromFlatArray(rawData,  [| rawData.Length |])

    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 2)
        for _ in 1 .. n do res <- dsharp.tensor(rawData)

#if !TINY
    //--------------------------------------------------------------
    // zeros

    [<Benchmark(Baseline=true); BenchmarkCategory("zeros")>]
    member perf.zeros_PyTorch() =  perf.simulatePythonResult("zeros")

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> torch.zeros([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Int64 -> torch.zeros([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Float32 -> torch.zeros([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Float64 -> torch.zeros([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res3 <- RawTensor.Zeros(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res <- dsharp.zeros( [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res3 <- RawTensor.Zeros(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res <- dsharp.zeros( [| perf.tensorSize |])

    //--------------------------------------------------------------
    // ones

    [<Benchmark(Baseline=true); BenchmarkCategory("ones")>]
    member perf.ones_PyTorch() = perf.simulatePythonResult("ones")

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> torch.ones([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Int64 -> torch.ones([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Float32 -> torch.ones([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Float64 -> torch.ones([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res3 <- RawTensor.Ones(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res <- dsharp.ones( [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res3 <- RawTensor.Ones(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res <- dsharp.ones( [| perf.tensorSize |])

    //--------------------------------------------------------------
    // rand

    [<Benchmark(Baseline=true); BenchmarkCategory("rand")>]
    member perf.rand_PyTorch() = perf.simulatePythonResult("rand")

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- 
                match dtype with 
                | Dtype.Int32 -> torch.randint(10L, [| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Int64 -> torch.randint(10L, [| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Float32 -> torch.rand([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | Dtype.Float64 -> torch.rand([| int64 perf.tensorSize |] , device=Device.Default.ToTorch)
                | _ -> failwith "unknown dtype in perf testing"

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- RawTensor.Random(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res <- dsharp.rand( [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- RawTensor.Random(Shape.create [| perf.tensorSize |])

    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res <- dsharp.rand( [| perf.tensorSize |])

    //--------------------------------------------------------------
    // addition

    [<Benchmark(Baseline=true); BenchmarkCategory("addition")>]
    member perf.addition_PyTorch() = perf.simulatePythonResult("addition")

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do 
            res4 <- torchTensor.add(torchTensor)

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res3 <- rawTensor.AddTT(rawTensor)

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10)
        for _ in 1 .. n do res <- dsharpTensor + dsharpTensor

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10)
        for _ in 1 .. n do res3 <- rawTensor.AddTT(rawTensor)

    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res <- dsharpTensor + dsharpTensor


    //--------------------------------------------------------------
    // addScalar

    [<Benchmark(Baseline=true); BenchmarkCategory("addScalar")>]
    member perf.addScalar_PyTorch() = perf.simulatePythonResult("addScalar")

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- torchTensor.add(torchScalar)

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- rawTensor.AddTT0(rawScalar)

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res <- dsharpTensor + dsharpScalarTensor

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- rawTensor.AddTT0(rawScalar)

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res <- dsharpTensor + dsharpScalarTensor

    //--------------------------------------------------------------
    // addWithAlpha

    [<Benchmark(Baseline=true); BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_PyTorch() = perf.simulatePythonResult("addWithAlpha")

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- torchTensor.add(torchTensor, alpha=torchScalar)

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res3 <- rawTensor.AddTT(rawTensor, alpha=rawScalar)

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res <- dsharpTensor.add(dsharpTensor.mul(dsharpScalar)) // TODO: no optimised routine in Tensor as yet

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res3 <- rawTensor.AddTT(rawTensor, alpha=rawScalar)

    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res <- dsharpTensor.add(dsharpTensor.mul(dsharpScalarTensor)) // TODO: no optimised routine in Tensor as yet

    //--------------------------------------------------------------
    // addInPlace

    [<Benchmark(Baseline=true); BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_PyTorch() = perf.simulatePythonResult("addInPlace")

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do 
            res4 <- torchTensor.add_(torchTensor)

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        res3 <- rawTensor.Clone()
        res3.SetMutable()
        for _ in 1 .. n do res3.AddInPlace(rawTensor)

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 10) 
        for _ in 1 .. n do res <- dsharpTensor + dsharpTensor // TODO: no optimised routine in RawTensor as yet

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        res3 <- rawTensor.Clone()
        res3.SetMutable()
        for _ in 1 .. n do res3.AddInPlace(rawTensor)

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 10) 
        for _ in 1 .. n do res <- dsharpTensor + dsharpTensor // TODO: no optimised routine in RawTensor as yet

    //--------------------------------------------------------------
    // matmul

    [<Benchmark(Baseline=true); BenchmarkCategory("matmul")>]
    member perf.matmul_PyTorch() : unit = perf.simulatePythonResult("matmul")

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_TorchSharp() = 
        let n = perf.configure(Backend.Torch, 1) 
        for _ in 1 .. n do 
            res4 <- matrixTorchTensor.matmul(matrixTorchTensor)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_RawTensor_Torch() = 
        let n = perf.configure(Backend.Torch, 1) 
        for _ in 1 .. n do res3 <- rawMatrixTensor.MatMulTT(rawMatrixTensor)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_Tensor_Torch() = 
        let n = perf.configure(Backend.Torch, 1) 
        for _ in 1 .. n do res <- dsharpMatrixTensor.matmul(dsharpMatrixTensor)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_RawTensor_Reference() = 
        let n = perf.configure(Backend.Reference, 1) 
        for _ in 1 .. n do res3 <- rawMatrixTensor.MatMulTT(rawMatrixTensor)

    [<Benchmark; BenchmarkCategory("matmul")>]
    member perf.matmul_Tensor_Reference() = 
        let n = perf.configure(Backend.Reference, 1) 
        for _ in 1 .. n do res <- dsharpMatrixTensor.matmul(dsharpMatrixTensor)

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


