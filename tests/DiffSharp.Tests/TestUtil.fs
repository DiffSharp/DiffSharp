namespace Tests

open System
open DiffSharp
open NUnit.Framework

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

    member _.full(length:int, value, ?device, ?backend, ?dtype) =
        dsharp.full(length, value, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.ones(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.ones(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.ones(length:int, ?device, ?backend, ?dtype) =
        dsharp.ones(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.zeros(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.zeros(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.zeros(length:int, ?device, ?backend, ?dtype) =
        dsharp.zeros(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.one(?device, ?backend, ?dtype) =
        dsharp.one(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.zero(?device, ?backend, ?dtype) =
        dsharp.zero(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.onehot(length, hot, ?device, ?backend, ?dtype) =
        dsharp.onehot(length, hot, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.arange(endVal:float, ?startVal:float, ?step:float, ?device, ?backend, ?dtype) =
        dsharp.arange(endVal, ?startVal=startVal, ?step=step, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member _.arange(endVal:int, ?startVal:int, ?step:int, ?device, ?backend, ?dtype) =
        dsharp.arange(endVal, ?startVal=startVal, ?step=step, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDtype)

    member c.arrayCreator1D(arr: double[]) =
        match c.dtype with 
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
        | Dtype.Float32 -> arr |> Array2D.map float32 :> Array
        | Dtype.Float64 -> arr |> Array2D.map double :> Array
        | Dtype.Byte -> arr |> Array2D.map byte :> Array
        | Dtype.Int8 -> arr |> Array2D.map int8 :> Array
        | Dtype.Int16 -> arr |> Array2D.map int16:> Array
        | Dtype.Int32 -> arr |> Array2D.map int32 :> Array
        | Dtype.Int64  -> arr |> Array2D.map int64 :> Array
        | Dtype.Bool -> arr |> Array2D.map (fun x -> abs x >= 1.0) :> Array

module Dtypes =

    // We run most tests at all these tensor types
    let Bool = [ Dtype.Bool ]
    let SignedIntegral = [ Dtype.Int8; Dtype.Int16; Dtype.Int32; Dtype.Int64 ]
    let UnsignedIntegral = [ Dtype.Byte ]
    let Integral = SignedIntegral @ UnsignedIntegral
    let FloatingPoint = [ Dtype.Float32; Dtype.Float64 ]
    let Float32 = [ Dtype.Float32 ]

    // Some operations have quirky behaviour on bool types, we pin these down manually
    let SignedIntegralAndFloatingPoint = FloatingPoint @ SignedIntegral
    let IntegralAndFloatingPoint = FloatingPoint @ Integral
    let IntegralAndBool = Integral @ Bool
    let All = FloatingPoint @ Integral @ Bool

module Combos =

    // Use these to experiment in your local branch
    //let backends = [ Backend.Reference ]
    //let backends = [ Backend.Torch ]
    //let backends = [ Backend.Reference; Backend.Torch; Backend.Register("TestDuplicate") ]
    //let backends = [ Backend.Reference; Backend.Torch ]
    //let backends = [ Backend.Reference; Backend.Register("TestDuplicate") ]
    //let backends = [ Backend.Register("TestDuplicate") ]
    //let getDevices _ = [ Device.CPU ]
    //let getDevices _ = [ Device.GPU ]
    
    //Use this in committed code
    let backends = [ Backend.Reference; Backend.Torch ]
    let getDevices (deviceType: DeviceType option, backend: Backend option) =
        dsharp.devices(?deviceType=deviceType, ?backend=backend)

    let makeCombos dtypes =
        [ for backend in backends do
            let ds = getDevices (None, Some backend)
            for device in ds do
              for dtype in dtypes do
                yield ComboInfo(defaultBackend=backend, defaultDevice=device, defaultDtype=dtype, defaultFetchDevices=getDevices) ]

    /// These runs though all devices, backends and various Dtype
    let Float32 = makeCombos Dtypes.Float32
    let Integral = makeCombos Dtypes.Integral
    let FloatingPoint = makeCombos Dtypes.FloatingPoint
    let UnsignedIntegral = makeCombos Dtypes.UnsignedIntegral
    let SignedIntegral = makeCombos Dtypes.SignedIntegral
    let SignedIntegralAndFloatingPoint = makeCombos Dtypes.SignedIntegralAndFloatingPoint
    let IntegralAndFloatingPoint = makeCombos Dtypes.IntegralAndFloatingPoint
    let Bool = makeCombos Dtypes.Bool
    let IntegralAndBool = makeCombos Dtypes.IntegralAndBool
    let All = makeCombos Dtypes.All

    /// This runs though all devices and backends but leaves the default Dtype
    let AllDevicesAndBackends = 
        [ for backend in backends do
            let ds = getDevices (None, Some backend)
            for device in ds do
              yield ComboInfo(defaultBackend=backend, defaultDevice=device, defaultFetchDevices=getDevices) ]

[<AutoOpen>]
module TestUtils =
    let isException f = Assert.Throws<Exception>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isInvalidOp f = Assert.Throws<InvalidOperationException>(TestDelegate(fun () -> f() |> ignore)) |> ignore

