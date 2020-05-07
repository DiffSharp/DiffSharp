namespace Tests

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open NUnit.Framework

// This captures the expected semantics of different DTypes
type ComboInfo(backend: Backend, device: Device, dtype: DType) =
    member _.backend = backend
    member _.device = device
    member _.dtype = dtype
    member _.tensor(data: obj) = dsharp.tensor(data, device=device, backend=backend, dtype=dtype)
    member _.arrayCreator1D(arr: double[]) =
        match dtype with 
        | DType.Float32 -> arr |> Array.map float32 :> Array
        | DType.Float64 -> arr |> Array.map double :> Array
        | DType.Int8 -> arr |> Array.map int8 :> Array
        | DType.Int16 -> arr |> Array.map int16:> Array
        | DType.Int32 -> arr |> Array.map int32 :> Array
        | DType.Int64  -> arr |> Array.map int64 :> Array
        | DType.Bool -> arr |> Array.map (fun x -> abs x >= 1.0) :> Array

    member _.arrayCreator2D(arr: double[,]) : Array =
        match dtype with 
        | DType.Float32 -> arr |> Array2D.map float32 :> Array
        | DType.Float64 -> arr |> Array2D.map double :> Array
        | DType.Int8 -> arr |> Array2D.map int8 :> Array
        | DType.Int16 -> arr |> Array2D.map int16:> Array
        | DType.Int32 -> arr |> Array2D.map int32 :> Array
        | DType.Int64  -> arr |> Array2D.map int64 :> Array
        | DType.Bool -> arr |> Array2D.map (fun x -> abs x >= 1.0) :> Array

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
    let backends = [ (* Backend.None; *) Backend.Torch ]
    let devices = [ (* Backend.None; *) Device.CPU ]

    // We run tests specific to floating point at these tensor types

    let makeCombos dtypes =
        [ for backend in backends do
          for device in devices do
          for dtype in dtypes do
          yield ComboInfo(backend, device, dtype) ]

    let Integral = makeCombos DTypes.Integral
    let FloatingPoint = makeCombos DTypes.FloatingPoint
    let IntegralAndFloatingPoint = makeCombos DTypes.IntegralAndFloatingPoint
    let Bool = makeCombos DTypes.Bool
    let IntegralAndBool = makeCombos DTypes.IntegralAndBool
    let All = makeCombos DTypes.All

[<AutoOpen>]
module TestUtils =
    let isException f = Assert.Throws<Exception>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isInvalidOp f = Assert.Throws<InvalidOperationException>(TestDelegate(fun () -> f() |> ignore)) |> ignore

[<TestFixture>]
type TestUtil () =

    [<SetUp>]
    member this.Setup () =
        ()

