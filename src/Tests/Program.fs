// Learn more about F# at http://fsharp.org

open System
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions

type Linear(inFeatures, outFeatures) =
    let w = Tensor.Random([inFeatures; outFeatures])
    let b = Tensor.Random([outFeatures])
    member l.Forward(value) = Tensor.MatMul(value, w) + b

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    DiffSharp.Seed(123)
    let layer = Linear(5, 3)
    let x = Tensor.Random([1; 5])
    let o = x |> layer.Forward |> Tensor.Relu
    printfn "%A" x
    printfn "%A" o
    
    0 // return an integer exit code
