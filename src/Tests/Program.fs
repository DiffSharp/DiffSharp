// Learn more about F# at http://fsharp.org

open System
open DiffSharp

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"
    let t = Tensor.Create(2.f)
    printfn "%A" t
    0 // return an integer exit code