// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.NN


type Model() =
    inherit Layer()
    let fc1 = Linear(2, 2)
    let fc2 = Linear(2, 1)
    do base.AddParameters(["fc1", fc1; "fc2", fc2])
    override l.Forward(x) =
        x 
        |> fc1.Forward |> Tensor.Relu
        |> fc2.Forward |> Tensor.Relu

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    // DiffSharp.Seed(123)
    // DiffSharp.NestReset()
    let model = Model()
    model.ReverseDiff()
    let x = Tensor.Random([1; 2])
    let o = model.Forward(x)
    printfn "%A" x
    printfn "%A" o
    printfn "%A\n\n" model.Parameters
    model.NoDiff()
    let x = Tensor.Random([1; 2])
    let o = model.Forward(x)
    printfn "%A" x
    printfn "%A" o
    printfn "%A\n\n" model.Parameters

    0 // return an integer exit code
