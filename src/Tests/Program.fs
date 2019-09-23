// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
// [<AbstractClass>]

// type Layer =
//     val Parameters: Dictionary<string, Tensor>
//     new(parameters) = {Parameters = parameters}
//     member l.ReverseDiff() = 
//         l.Parameters.Keys |> Seq.iter (fun t -> l.Parameters.[t] <- l.Parameters.[t].ReverseDiff())

// type Linear =
//     inherit Layer
//     new(inFeatures, outFeatures) = 
//         let w = Tensor.Random([inFeatures; outFeatures])
//         let b = Tensor.Random([outFeatures])

//         let d = Dictionary()
//         d.Add("W", w)
//         d.Add("b", b)
//         {inherit Layer(Dictionary(d))}

//     // Parameters <- Dictionary()
//         // let d = Dictionary()
//         // d.Add("W", w)
//         // d.Add("b", b)
//         // d

type Layer(parameters) =
    member val Parameters:Dictionary<string, Tensor> = Dictionary(parameters |> Map.ofList)
    member l.ReverseDiff() = 
        let keys = Array.create l.Parameters.Count ""
        l.Parameters.Keys.CopyTo(keys, 0)
        for k in keys do l.Parameters.[k] <- l.Parameters.[k].ReverseDiff()

type Linear(inFeatures, outFeatures) =
    inherit Layer([
        "W", Tensor.Random([inFeatures; outFeatures]); 
        "b", Tensor.Random([outFeatures])])
    do printfn ""
    member l.Forward(value) = Tensor.MatMul(value, l.Parameters.["W"]) + l.Parameters.["b"]

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    DiffSharp.Seed(123)
    DiffSharp.NestReset()
    let layer = Linear(5, 3)
    let x = Tensor.Random([1; 5])

    layer.ReverseDiff()
    let o = x |> layer.Forward |> Tensor.Relu
    // printfn "%A" layer.Parameters
    printfn "%A" x
    printfn "%A" o
    o.Reverse(Tensor.Random([1;3]))
    printfn ""
    // printfn "%A" layer.Parameters
    printfn "%A" x
    printfn "%A" o
    
    0 // return an integer exit code
