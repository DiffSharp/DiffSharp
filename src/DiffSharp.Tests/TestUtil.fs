namespace Tests

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open NUnit.Framework

// This captures the expected semantics of different DTypes
type ComboInfo(?defaultBackend: Backend, ?defaultDevice: Device, ?defaultDType: DType) =

    let dflt x y = match x with Some x -> Some x | None -> y

    member _.backend = defaultArg defaultBackend Backend.Default

    member _.device = defaultArg defaultDevice Device.Default

    member _.dtype = defaultArg defaultDType DType.Default

    member _.tensor(data: obj, ?device, ?backend, ?dtype) =
        dsharp.tensor(data, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.randn(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.randn(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.randn(length:int, ?device, ?backend, ?dtype) =
        dsharp.randn(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.rand(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.rand(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.rand(length:int, ?device, ?backend, ?dtype) =
        dsharp.rand(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.full(shape:seq<int>, value, ?device, ?backend, ?dtype) =
        dsharp.full(shape, value, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.full(length:int, value, ?device, ?backend, ?dtype) =
        dsharp.full(length, value, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.ones(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.ones(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.ones(length:int, ?device, ?backend, ?dtype) =
        dsharp.ones(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.zeros(shape:seq<int>, ?device, ?backend, ?dtype) =
        dsharp.zeros(shape, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.zeros(length:int, ?device, ?backend, ?dtype) =
        dsharp.zeros(length, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.one(?device, ?backend, ?dtype) =
        dsharp.one(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.zero(?device, ?backend, ?dtype) =
        dsharp.zero(?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.onehot(length, hot, ?device, ?backend, ?dtype) =
        dsharp.onehot(length, hot, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member _.arange(endVal, ?device, ?backend, ?dtype) =
        dsharp.arange(endVal, ?device=dflt device defaultDevice, ?backend=dflt backend defaultBackend, ?dtype=dflt dtype defaultDType)

    member c.arrayCreator1D(arr: double[]) =
        match c.dtype with 
        | DType.Float32 -> arr |> Array.map float32 :> Array
        | DType.Float64 -> arr |> Array.map double :> Array
        | DType.Int8 -> arr |> Array.map int8 :> Array
        | DType.Int16 -> arr |> Array.map int16:> Array
        | DType.Int32 -> arr |> Array.map int32 :> Array
        | DType.Int64  -> arr |> Array.map int64 :> Array
        | DType.Bool -> arr |> Array.map (fun x -> abs x >= 1.0) :> Array
        | DType.Other _ -> failwith "unexpected user-defined type"

    member c.arrayCreator2D(arr: double[,]) : Array =
        match c.dtype with 
        | DType.Float32 -> arr |> Array2D.map float32 :> Array
        | DType.Float64 -> arr |> Array2D.map double :> Array
        | DType.Int8 -> arr |> Array2D.map int8 :> Array
        | DType.Int16 -> arr |> Array2D.map int16:> Array
        | DType.Int32 -> arr |> Array2D.map int32 :> Array
        | DType.Int64  -> arr |> Array2D.map int64 :> Array
        | DType.Bool -> arr |> Array2D.map (fun x -> abs x >= 1.0) :> Array
        | DType.Other _ -> failwith "unexpected user-defined type"

module DTypes =

    // We run most tests at all these tensor types
    let Bool = [ DType.Bool ]
    let Integral = [DType.Int8; DType.Int16; DType.Int32; DType.Int64]
    let FloatingPoint = [DType.Float32; DType.Float64]

    // Some operations have quirky behaviour on bool types, we pin these down manually
    let IntegralAndFloatingPoint = FloatingPoint @ Integral
    let IntegralAndBool = Integral @ Bool
    let All = FloatingPoint @ Integral @ Bool

module Combos =

    let backends = [ Backend.None; Backend.Register("TestDuplicate") ]

    let devices = [ Device.CPU ]

    let makeCombos dtypes =
        [ for backend in backends do
            for device in devices do
              for dtype in dtypes do
                yield ComboInfo(backend, device, dtype) ]

    /// These runs though all devices, backends and DType
    let Integral = makeCombos DTypes.Integral
    let FloatingPoint = makeCombos DTypes.FloatingPoint
    let IntegralAndFloatingPoint = makeCombos DTypes.IntegralAndFloatingPoint
    let Bool = makeCombos DTypes.Bool
    let IntegralAndBool = makeCombos DTypes.IntegralAndBool
    let All = makeCombos DTypes.All

    /// This runs though all devices and backends but leaves the default DType
    let AllDevicesAndBackends = 
        [ for backend in backends do
          for device in devices do
          yield ComboInfo(defaultBackend=backend, defaultDevice=device) ]

[<AutoOpen>]
module TestUtils =
    let isException f = Assert.Throws<Exception>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isInvalidOp f = Assert.Throws<InvalidOperationException>(TestDelegate(fun () -> f() |> ignore)) |> ignore

[<TestFixture>]
type TestUtil () =

    [<SetUp>]
    member this.Setup () =
        ()

