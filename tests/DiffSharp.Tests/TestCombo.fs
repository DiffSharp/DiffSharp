// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open DiffSharp

// This captures the expected semantics of different Dtypes
type ComboInfo(?defaultBackend: Backend, ?defaultDevice: Device, ?defaultDtype: Dtype, ?defaultFetchDevices: (DeviceType option * Backend option -> Device list)) =

    let dflt x y = match x with Some x -> Some x | None -> y

    member _.backend = defaultArg defaultBackend Backend.Default

    member _.device = defaultArg defaultDevice Device.Default

    member _.devices(?deviceType, ?backend) = 
       let f = defaultArg defaultFetchDevices (fun (deviceType, backend) -> dsharp.devices(?deviceType=deviceType, ?backend=backend))
       f (deviceType, backend)

    member _.dtype = defaultArg defaultDtype Dtype.Default
    
    member _.tensor(data: obj, ?device, ?backend, ?dtype) =
        dsharp.tensor(data, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.randn(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.randn(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.randn(length:int, ?device, ?backend, ?dtype) =
        dsharp.randn(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.rand(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.rand(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.rand(length:int, ?device, ?backend, ?dtype) =
        dsharp.rand(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.randint(low:int, high:int, shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.randint(low, high, shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.randint(low:int, high:int, length:int, ?device, ?backend, ?dtype) =
        dsharp.randint(low, high, length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.full(shape:seq<int>, value, ?device, ?backend, ?dtype) =
        dsharp.full(shape, value, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.full(length:int, value:scalar, ?device, ?backend, ?dtype) =
        dsharp.full(length, value, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.ones(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.ones(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.ones(length:int, ?device, ?backend, ?dtype) =
        dsharp.ones(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.zeros(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.zeros(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.zeros(length:int, ?device, ?backend, ?dtype) =
        dsharp.zeros(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.empty(?device, ?backend, ?dtype) =
        dsharp.empty(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.empty(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.empty(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.empty(length:int, ?device, ?backend, ?dtype) =
        dsharp.empty(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.one(?device, ?backend, ?dtype) =
        dsharp.one(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.zero(?device, ?backend, ?dtype) =
        dsharp.zero(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.move(tensor, ?device, ?backend, ?dtype) =
        dsharp.move(tensor, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.onehot(length, hot, ?device, ?backend, ?dtype) =
        dsharp.onehot(length, hot, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.eye(rows:int, ?cols:int, ?device, ?backend, ?dtype) =
        dsharp.eye(rows, ?cols=cols, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.arange(endVal:float, ?startVal:float, ?step:float, ?device, ?backend, ?dtype) =
        dsharp.arange(endVal, ?startVal=startVal, ?step=step, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.arange(endVal:int, ?startVal:int, ?step:int, ?device, ?backend, ?dtype) =
        dsharp.arange(endVal, ?startVal=startVal, ?step=step, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.linspace(startVal:float, endVal:float, steps:int, ?device, ?backend, ?dtype) =
        dsharp.linspace(startVal, endVal, steps, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.linspace(startVal:int, endVal:int, steps:int, ?device, ?backend, ?dtype) =
        dsharp.linspace(startVal, endVal, steps, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.logspace(startVal:float, endVal:float, steps:int, ?baseVal, ?device, ?backend, ?dtype) =
        dsharp.logspace(startVal, endVal, steps, ?baseVal=baseVal, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.logspace(startVal:int, endVal:int, steps:int, ?baseVal, ?device, ?backend, ?dtype) =
        dsharp.logspace(startVal, endVal, steps, ?baseVal=baseVal, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member c.arrayCreator1D(arr: double[]) =
        match c.dtype with 
        | Dtype.Float16 -> arr |> Array.map float32 :> Array
        | Dtype.BFloat16 -> arr |> Array.map float32 :> Array
        | Dtype.Float32 -> arr |> Array.map float32 :> Array
        | Dtype.Float64 -> arr |> Array.map double :> Array
        | Dtype.Byte -> arr |> Array.map byte :> Array
        | Dtype.Int8 -> arr |> Array.map int8 :> Array
        | Dtype.Int16 -> arr |> Array.map int16:> Array
        | Dtype.Int32 -> arr |> Array.map int32 :> Array
        | Dtype.Int64  -> arr |> Array.map int64 :> Array
        | Dtype.Bool -> arr |> Array.map (fun x -> abs x >= 1.0) :> Array

    member c.arrayCreator2D(arr: double[,]) : Array =
        match c.dtype with 
        | Dtype.BFloat16 -> arr |> Array2D.map float32 :> Array
        | Dtype.Float16 -> arr |> Array2D.map float32 :> Array
        | Dtype.Float32 -> arr |> Array2D.map float32 :> Array
        | Dtype.Float64 -> arr |> Array2D.map double :> Array
        | Dtype.Byte -> arr |> Array2D.map byte :> Array
        | Dtype.Int8 -> arr |> Array2D.map int8 :> Array
        | Dtype.Int16 -> arr |> Array2D.map int16:> Array
        | Dtype.Int32 -> arr |> Array2D.map int32 :> Array
        | Dtype.Int64  -> arr |> Array2D.map int64 :> Array
        | Dtype.Bool -> arr |> Array2D.map (fun x -> abs x >= 1.0) :> Array

