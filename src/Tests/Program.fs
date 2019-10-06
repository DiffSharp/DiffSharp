// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.NN
open DiffSharp.RawTensor


type Model() =
    inherit Layer()
    let fc1 = Linear(2, 16)
    let fc2 = Linear(16, 1)
    do base.AddParameters(["fc1", fc1])
    override l.Forward(x) =
        x 
        |> fc1.Forward |> Tensor.Relu
        |> fc2.Forward |> Tensor.Relu

let optimize (model:Layer) (lr:Tensor) =
    let update k (p:Parameter) = 
        // printfn "updating %A" k; 
        p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative
    model.Map(update)

[<AutoOpen>]
module ExtraPrimitives =
    let inline tryUnbox<'a> (x:obj) =
        match x with
        | :? 'a as result -> Some (result)
        | _ -> None


    // if dim = 0 then
    //     let mutable s = Tensor.ZerosLike(t).[0]
    //     for i=0 to t.Shape.[0]-1 do
    //         s <- s + t.[i]
    //     s
    // elif dim = 1 then
    //     let mutable s = Tensor.ZerosLike(t).[*,0]
    //     for i=0 to t.Shape.[1]-1 do
    //         s <- s + t.[*,i]
    //     s
    // else
    //     failwith "Not implemented"

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    // DiffSharp.Seed(125)
    // DiffSharp.NestReset()
    // let model = Model()
    // model.ReverseDiff()
    // let data = Tensor.Create([[0.;0.;0.];[0.;1.;1.];[1.;0.;1.];[1.;1.;0.]])
    // let x = data.[*,0..1]
    // let y = data.[*,2..]
    // printfn "%A" x
    // printfn "%A" y

    // let mseloss (x:Tensor) (y:Tensor) = Tensor.Sum((x - y) * (x - y)) / x.Shape.[0]

    // for i=0 to 10000 do    
    //     model.ReverseDiff()
    //     let o = model.Forward(x).View([4])
    //     let loss = mseloss o y
    //     printfn "prediction: %A, loss: %A" (o.NoDiff()) (loss.NoDiff())
    //     // printfn "%A" loss
    //     loss.Reverse()
    //     optimize model (Tensor.Create(0.01))

    let x = Tensor.Create([[[0.3787;0.7515;0.2252;0.3416];
          [0.6078;0.4742;0.7844;0.0967];
          [0.1416;0.1559;0.6452;0.1417]];
 
         [[0.0848;0.4156;0.5542;0.4166];
          [0.5187;0.0520;0.4763;0.1509];
          [0.4767;0.8096;0.1729;0.6671]]])
    let x0 = x.Variance(0)
    let x1 = x.Variance(1)
    let x2 = x.Variance(2)
    printfn "%A" x
    printfn "\n%A" x0
    printfn "\n%A" x1
    printfn "\n%A" x2
    0 // return an integer exit code
