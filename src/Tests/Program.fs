// Learn more about F# at http://fsharp.org

open System
open DiffSharp
open TorchSharp.Tensor

[<EntryPoint>]
let main argv =
    printfn "DiffSharp tests"
    let t = TorchSharp.Tensor.FloatTensor.From(1.f)
    printfn "%A" t

    let tt = Tensor.Create(2.f)
    printfn "%A" tt
    0 // return an integer exit code